import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.info()
data.head(10)
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="white", fmt= '.1f',ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(15, 5))

sns.countplot(data['GRE Score'])

plt.xticks(rotation= 70)

plt.show()



f,ax = plt.subplots(figsize=(15, 5))

sns.countplot(data['TOEFL Score'])

plt.xticks(rotation= 70)

plt.show()
from sklearn.linear_model import LinearRegression #add linear regression library

reg = LinearRegression()

x = data.iloc[:,1].values.reshape(-1,1)

y = data.iloc[:,8].values.reshape(-1,1)

reg.fit(x,y) #create line

y_head = reg.predict(x) #predict chance of admit

b0 = reg.intercept_ 

b1 = reg.coef_

print("B0 :",b0)

print("B1 :",b1)



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Values :",r2_score(y,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good. 



#visualization

f,ax = plt.subplots(figsize=(10, 10))

plt.scatter(data['GRE Score'],data.iloc[:,-1], color="red")

plt.plot(x,y_head, color="blue")

plt.show()
from sklearn.linear_model import LinearRegression #add linear regression library

reg = LinearRegression()

x = data.iloc[:,2].values.reshape(-1,1)

y = data.iloc[:,8].values.reshape(-1,1)

reg.fit(x,y) #create line

y_head = reg.predict(x) #predict chance of admit

b0 = reg.intercept_

b1 = reg.coef_

print("B0 :",b0)

print("B1 :",b1)



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Value :",r2_score(y,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good. 



#visualization

f,ax = plt.subplots(figsize=(10, 10))

plt.scatter(data['TOEFL Score'],data.iloc[:,-1], color="red")

plt.plot(x,y_head)

plt.show()
from sklearn.linear_model import LinearRegression #add linear regression library

reg = LinearRegression()

x = data.iloc[:,6].values.reshape(-1,1)

y = data.iloc[:,8].values.reshape(-1,1)

reg.fit(x,y)#create line

y_head = reg.predict(x) #predict chance of admit

b0 = reg.intercept_

b1 = reg.coef_ 

print("B0 :",b0)

print("B1 :",b1)



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Value :",r2_score(y,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good. 



#visualization

f,ax = plt.subplots(figsize=(10,10))

plt.scatter(data['CGPA'],data.iloc[:,-1], color="red")

plt.plot(x,y_head, color="blue")

plt.show()
from sklearn.linear_model import LinearRegression #add linear regression library

reg = LinearRegression()

x = data.iloc[:,[1,2,6]].values

y = data.iloc[:,8].values.reshape(-1,1)

reg.fit(x,y) #create line

y_head = reg.predict(x) #predict chance of admit

print("B0 :",reg.intercept_)

print("B1,B2 and B3 :",reg.coef_)



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Value :",r2_score(y,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good.
from sklearn.preprocessing import PolynomialFeatures # add Polynomial and Linear Regression library

from sklearn.linear_model import LinearRegression

polynomial_reg = PolynomialFeatures(degree=4) #set degree

reg = LinearRegression()

x = data.iloc[:,6].values.reshape(-1,1)

y = data.iloc[:,8].values.reshape(-1,1)

x_polynomial = polynomial_reg.fit_transform(x) # Get the degree and add

reg.fit(x_polynomial,y) #create line 

y_head = reg.predict(x_polynomial) # predict chance of admit

print("B0 :",reg.intercept_)

print("B1,B2,B3,B4 and B5 :",reg.coef_)



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Value :",r2_score(y,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good.



#visualization

f,ax = plt.subplots(figsize=(10,10))

plt.scatter(data['CGPA'],data.iloc[:,-1], color="red")

plt.plot(x,y_head, color="blue")

plt.show()
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

x = data.iloc[:,1].values.reshape(-1,1)

y = data.iloc[:,8].values.reshape(-1,1)

tree_reg.fit(x,y) # create model



min_x = min(data.iloc[:,1].values)

max_x = max(data.iloc[:,1].values)

array = np.arange(min_x,max_x,0.1).reshape(-1,1)

y_head = tree_reg.predict(array) # predict chance of admit



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Value :",r2_score(y,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good.



#visualization

f,ax = plt.subplots(figsize=(15,10))

plt.scatter(x,y,color="red",alpha=0.4)

plt.plot(array,y_head,color="green")

plt.show()
from sklearn.ensemble import RandomForestRegressor # add Random Forest Regression library

data_drop = data.drop(['Chance of Admit '],axis=1)

data_change_of_admit = data.iloc[:,-1].values.reshape(-1,1)



reg = RandomForestRegressor(n_estimators=100, random_state = 42) # determine tree count and random data count

reg.fit(data_drop,data_change_of_admit) # create model

y_head = reg.predict(data_drop) # predict chance of admit



#Evalutaion Regression Model

from sklearn.metrics import r2_score

print("R Square Value :",r2_score(data_change_of_admit,y_head)) #Evaluation algorithm, If it is close to 1, Model is so good.




