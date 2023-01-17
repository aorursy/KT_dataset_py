# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:02:11 2020

@author: Elif
"""
#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data

df=pd.read_csv("/kaggle/input/verisetim.csv",sep=";")
df.head()

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#linear reggesion
#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg=LinearRegression()
x=df.deneyim.values.reshape(-1,1);
y=df.maas.values.reshape(-1,1);
linear_reg.fit(x,y) #linefit
#prediction
import numpy as np
b0=linear_reg.predict([[0]])
print("b0:",b0)
b0=linear_reg.intercept_
print("b0:",b0)
b1=linear_reg.coef_
print("b1:",b1)
maas_yeni=1663+1138*11
print(maas_yeni)
print(linear_reg.predict([[11]]))
#visualize line
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).reshape(-1,1) #deneyim

plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(array) #maas

plt.plot(array, y_head,color="red")