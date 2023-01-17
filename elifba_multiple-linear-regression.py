# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:00:16 2020

@author: Elif
"""
#import library
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#import data
df=pd.read_csv("/kaggle/input/multiplelineardata.csv",sep=";");
df.head()


x=df.iloc[:,[0,2]].values #independent variables
y=df.maas.values.reshape(-1,1) #dependent variable

multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

#intercept
print("b0:",multiple_linear_regression.intercept_)
print("b1,b2:",multiple_linear_regression.coef_)

#predict
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))