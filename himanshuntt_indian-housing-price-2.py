#importing library

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as seabornInstance
#importing data set

dataset=pd.read_csv('../input/houseprice.csv')

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values
plt.figure(figsize=(6,4))

plt.tight_layout()

seabornInstance.distplot(y)
#Splitting the data 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state = 0)
#Training

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train)
#intercept and coff

y_predict=regressor.predict(x_test)

print(regressor.intercept_)

print(regressor.coef_)
#prediction

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})

df
#Actual vs prediction plotting

df1 = df.head(25)

df1.plot(kind='bar',figsize=(8,3))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#Accuracy

import sklearn

from sklearn import metrics

print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test, y_predict))  

print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y_test, y_predict))  

print('Root Mean Squared Error:', np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_predict)))