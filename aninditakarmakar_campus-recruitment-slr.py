#Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Importing the dataset

dataset = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
dataset.head(10)
dataset.info()
# Checking whether the outcome and predictor columns have null values

print(dataset['degree_p'].isnull().values.any())

print(dataset['mba_p'].isnull().values.any())
# Splitting the dataset into outcome and predictor variable

x = dataset.iloc[:,7].values

y = dataset.iloc[:,12].values
print(x)
print(y)
## Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
## Training the Simple Linear Regression model on the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(np.reshape(x_train,(len(x_train),1)),np.reshape(y_train,(len(y_train),1)))
## Predicting the test results

y_pred = regressor.predict(np.reshape(x_test,(len(x_test),1)))
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#Visualising the training set results

plt.style.use('classic')

plt.figure(figsize=(8,6))

plt.scatter(x_train,y_train,color='blue')

plt.plot(x_train,regressor.predict(np.reshape(x_train,(len(x_train),1))),color='lawngreen',linewidth=3)

plt.title('Degree % Vs MBA % (Training set)')

plt.xlabel("Predictor variable- Degree %")

plt.ylabel("Outcome variable- MBA %")

plt.show()
## Calculating r2 score for training set

from sklearn.metrics import r2_score

r2_score(y_train,regressor.predict(np.reshape(x_train,(len(x_train),1))))
#Visualising the test set results

plt.style.use('classic')

plt.figure(figsize=(8,6))

plt.scatter(x_test,y_test,color='blue')

plt.plot(x_test,y_pred,color='lawngreen',linewidth=3)

plt.title('Degree % Vs MBA % (Test set)')

plt.xlabel("Predictor variable- Degree %")

plt.ylabel("Outcome variable- MBA %")

plt.show()
## Calculating r2 score for test set

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
print(regressor.coef_)
print(regressor.intercept_)