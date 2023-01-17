import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plotting data

import seaborn as sns # Advanced visualization
excel_file = '/kaggle/input/linear-regression/linear.xlsx'

dataset = pd.read_excel(excel_file)
dataset
dataset.iloc[:,:-1]
#convert into array



x=dataset.iloc[:,:-1].values
x
y=dataset.iloc[:,1].values
y
from sklearn.model_selection import train_test_split  

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state=0) 
x_train
x_test
y_train
y_test
from sklearn.linear_model import LinearRegression

regressor =  LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
#scope of the line



regressor.coef_
#intercept of line 

regressor.intercept_