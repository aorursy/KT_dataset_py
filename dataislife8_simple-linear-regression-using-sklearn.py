import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.linear_model import LinearRegression
#data = pd.read_csv(r'S:\Udemy\Data Science Bootcamp\Data samples for practice csv_files\1.01. Simple linear regression.csv')

data = pd.read_csv('/kaggle/input/gpa-prediction-using-sat-scores/1.01. Simple linear regression.csv')

data.head()
y = data['GPA']     # -> Dependent variable (TO BE PREDICTED)

x = data['SAT']     # -> Independent variable
x.shape, y.shape
reg = LinearRegression()
#reg.fit(x,y)  

'''This will give error as the shape shown above is 1-D, Now we have to re shape it to 2-D'''
x_matrix = x.values.reshape(-1,1)

x_matrix.shape  #Turned to 2-D
reg.fit(x_matrix,y)
reg.score(x_matrix,y)
reg.coef_
reg.intercept_
plt.scatter(x,y)

yhat = reg.coef_*x_matrix + reg.intercept_

plt.plot(x,yhat, lw=3, color='red')

plt.xlabel('SAT', fontsize=15)

plt.ylabel('GPA', fontsize=15)