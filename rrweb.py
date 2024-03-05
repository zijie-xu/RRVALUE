# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:30:52 2024

@author: zijie.xu
"""
import sklearn
from sklearn.ensemble import RandomForestRegressor  
import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
import pickle
import os

import requests
url = 'https://gitee.com/zijiexu/work/raw/master/rrmodel.dat'
r=requests.get(url)
with open('rr1.dat','wb') as f:
    f.write(r.content)
  #  f.close
current_dir = os.getcwd()
full_path = os.path.join(current_dir, 'rr1.dat')
#rrmodelfile = r'C:\Users\zijie.xu\Desktop\2024年\01 Python\03 rr预测\rrmodel.dat'
load_model=pickle.load(open(full_path,"rb"))
url1 = 'https://gitee.com/zijiexu/work/raw/master/2.jpg'
r1=requests.get(url1)
with open('rr.jpg','wb') as f1:
    f1.write(r1.content)
  #  f.close
full_path1 = os.path.join(current_dir, 'rr.jpg')


st.title("Tire RR Values Prediction")
row1 = st.columns(2)
row2 = st.columns(2)
row3 = st.columns(2)
input_data=pd.DataFrame()
output_data=pd.DataFrame()
df=pd.DataFrame()

rmse=0
rsq=0

with row1[0]:
    with st.container():
        uploaded_file = st.file_uploader("Choose a file",label_visibility="collapsed")
        if uploaded_file is not None:  
            # Can be used wherever a "file-like" object is accepted:
            input_data = pd.read_csv(uploaded_file,delimiter='	')
            inputdata=np.array(input_data)
            output_data=[load_model.predict(inputdata),load_model.predict(inputdata)/inputdata[:,-2]*1000]
            output_data=np.array(output_data).T
    with st.container():
        st.dataframe(input_data)
with row1[1]:
    with st.container():
        st.write('RMSE(均方根误差）：%.2f'%rmse)
        st.write('模型测试值与预测值相关性系数R2：%.2f'%rsq)
        if st.button('滚动阻力预测'):
            
            df = pd.DataFrame(output_data,columns=('Fx, N','RR, N/kN'))
    with st.container():
        st.dataframe(df)
with row2[0]:
    with st.container():
        IMG = Image.open(full_path1)
        st.image(IMG)
with row2[1]:
    with st.container():
        st.write('免责声明：')
        st.write('1.本模型适用规格为R16、R17和R18；')
        st.write('2.建模的基础数据均为非液体黄金胶种；')
        st.write('3.模型的RMSE使用15组数据进行计算（每个规格各5组）；')
        st.write('4.此预测结果供使用人员筛选方案与结果预测使用；对于个别异常数据请及时与秦海兴直接联系。')
        st.write('版本号：RRP.1.1.1')
    
        
