# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("i am implementing basic linear regression model on canada per capita income to practice as a begginer")

print("Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of independent variables being used.")

print("Goal -  Predict canada's per capita income in year 2020")
#importing data-set using pandas library

import pandas as pd #pandas is a library ,but rather than using the name pandas , it's instructed to use the name pd instead.

df = pd.read_csv("../input/canada-per-capita-income-single-variable-data-set/canada_per_capita_income.csv")

df
#now we  find how many rows and columns we have in our data-set

df.shape
#checking keys

df.keys()
df.columns = ['year', 'income']

df

df.columns
#y = m*x + c     this is our straight line equation  where y is the dependent variable, x is the independent variable, m is the slope of the line and c is y-intercept

#y = income (dependent data)

#x = year(independent data)
x = df.year

y = df.income

y
#to find m and c values , we have one library sklearn library through this we can import pre-builded function Linear Regression
#firstly before using our independent variable year we have to convert it into 2-D array because iin fit fction it is compulsory to use independent variable in 2-D array form

x.values   #see this is in 1-D array form
x = df.iloc[:, 0:1].values  #converting x into 2-D array

y = df.iloc[:, 1].values

x

y
from sklearn.linear_model import LinearRegression
model = LinearRegression()   #this model variable will be like our machine brain which has to be trained by the straight line formula

model.fit(x,y)  #universal fubnction by which machine will train automatically
#now our machine knows everything about slope , intercept etc.

#its time to find m and c values

m =  model.coef_    #machine knows this

c = model.intercept_

m
c
y = (m*x) + c

y
y_predict = model.predict(x)   #predicted value of y replaced formula mx+c by fuction

y_predict

new_year = 2020

new_income = model.predict([[new_year]])

new_income
#plotting graphs 

import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.scatter([new_year],new_income, color = "yellow")

plt.plot(x,y_predict,color="red")

plt.show