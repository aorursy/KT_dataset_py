import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import sklearn
oecd_bli=pd.read_csv('E:/data/oecd_bli_2017.csv',thousands=',')
gdp_per_capita=pd.read_csv('E:/data/gdp_per_capita.csv',thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")
country_stats=(oecd_bli,gdp_per_capita)
x=np.c_[country_stats["GDP per capita"]]

y=np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind='scatter',x="GDP per capita",y='Life satisfaction') 

plt.show()
lin_reg_model=sklearn.linear_model.LinearRegression()
lin_reg_model.fit(X,y)
#Make a prediction for Cyprus

X_new=[[22587]] #cyprus'GDP per capita print(lin_reg_model.predict(X_new)