# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline




#Reading the data
data=pd.read_csv('../input/position-salaries/Position_Salaries.csv')
data.head()
#Finding out the shape of the data
data.shape
#Defining the respective input and ouputs
X=data.iloc[:,1:2].values
y=data.iloc[:,-1].values
#Finding out the shape of X
X.shape
#Finding out the shape og y
y.shape
#plotting the graph to find out the nature of the data
plt.scatter(data.Level,data.Salary)
#importing the required libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#Defining a function for polynomial regression
def polynomialRegression(X,y,k=14):

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)#Splitting the data into training and testing data

  poly = PolynomialFeatures(degree=k)#Creating an object for the class PolynomialFeatures
  X_poly = poly.fit_transform(X_train)#Fitting the training data
  lr = LinearRegression()#Creating an object for LinearRegression class
  lr.fit(X_poly,y_train)
  
  X_test_poly =poly.fit_transform(X_test)
  y_pred=lr.predict(X_test_poly)#Predicting the output

  training_score = r2_score(y_train, lr.predict(X_poly))
  test_score = r2_score(y_test,y_pred)#Finding out the accuracy of the model
  
  return training_score, test_score
#Function to find out the accurate value of k so that we get a correct value of r2_score
train=[]
test=[]
for i in range(1,10):
  r2train,r2test=polynomialRegression(X,y,k=i)
  train.append(r2train)
  test.append(r2test)
x=np.arange(9)+1
plt.plot(x,train,label="Training")
plt.plot(x,test,label="Test")
plt.legend()
plt.xlabel("Value of k")
plt.ylabel("r2-Score")
plt.title("R2-Score");
plt.show()

#Function to plot the best fit line
def polynomialRegression(X,y,k):

    poly = PolynomialFeatures(degree=k)
    X_poly = poly.fit_transform(X)
    lr = LinearRegression()
    lr.fit(X_poly,y)
  
    X_test_poly =poly.fit_transform(X)
    y_pred=lr.predict(X_test_poly)
    
    plt.plot(X,y_pred, label="Model",color='red')
    plt.scatter(X, y, label="data",color='blue')
    plt.legend()
    plt.show()
    
    print("The accuracy of the model is " ,r2_score(y,y_pred)*100)   
polynomialRegression(X,y,4)
polynomialRegression(X,y,3)
polynomialRegression(X,y,2)
