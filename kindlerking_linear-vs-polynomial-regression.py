# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/position-salaries/Position_Salaries.csv")
df.info()
#LINEAR REGRESSION
x=df.iloc[:,1:2].values
y=df.iloc[:,2:3].values
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
l=LinearRegression()
l.fit(x,y)
Salary_pred=l.predict(x)
df["Salary_pred"]=Salary_pred
df
#we are very off from the salary at first glance itself

#EVALUATING WITH METRICS
rmse=np.sqrt(mean_squared_error(y,Salary_pred))
R2=r2_score(y,Salary_pred)
arr={'linear_rmse':rmse,'linear_R2':R2}
arr
#Linear Regression fails severely in these cases and Polynomial Regression comes to the rescue.
plt.scatter(x,y)
plt.plot(x,Salary_pred,color='Red')
#POLYNOMIAL REGRESSION 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
poly_feat=PolynomialFeatures(degree=2)
x_poly=poly_feat.fit_transform(x)
l=LinearRegression()
l.fit(x_poly,y)
y_polypred=l.predict(x_poly)
plt.scatter(x,y)
plt.plot(x,y_polypred,color='r')
#CALCULATING ERROR TO COMPARE LINEAR AND POLYNOMIAL
polyrmse=np.sqrt(mean_squared_error(y,y_polypred))
polyR2=r2_score(y,y_polypred)
arr1={'Poly_rmse':polyrmse,'polyR2':polyR2}
arr.update(arr1)
arr
