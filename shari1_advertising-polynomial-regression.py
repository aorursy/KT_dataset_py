# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Getting the data



data=pd.read_csv('../input/Advertising.csv')



data.head()

data=data.drop(['Unnamed: 0'],axis=1)



sns.pairplot(data)



sns.pairplot(data=data,x_vars=['TV','radio','newspaper'],y_vars='sales',kind='reg')

data.isnull().sum()

#No null values in the dataset



data.corr()





#Splitting the data

X=data.iloc[:,:-1]

y=data['sales']



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)



#Linear Regression



#Training the model

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)





#Evaluation of test set

y_pred=lr.predict(X_test)

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score



print('Accuracy of training set',lr.score(X_train,y_train))



print('Root Mean Square Error is',np.sqrt(mean_squared_error(y_test,y_pred)))

print('R2 score is',r2_score(y_test,y_pred))





#Now if we remove newspaper feature 



X=data.drop(['newspaper','sales'],axis=1)

y=data['sales']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)

lr.fit(X_train,y_train)

y_pred1=lr.predict(X_test)



print('Accuracy of training set',lr.score(X_train,y_train))



print('RMSE is',np.sqrt(mean_squared_error(y_test,y_pred1)))

print('R2 score is',r2_score(y_test,y_pred1))



#Using Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures



def create_polynomial(degree):

    poly_features=PolynomialFeatures(degree)

    X_train_poly=poly_features.fit_transform(X_train)

    lr_poly=LinearRegression()

    lr_poly.fit(X_train_poly,y_train)

    

    #Predicting on train data

    y_train_poly=lr_poly.predict(X_train_poly)

    

    #Predicting on test data

    y_test_poly=lr_poly.predict(poly_features.fit_transform(X_test))

    

    #Evaluation on train data

    print('RMSE for train data is',np.sqrt(mean_squared_error(y_train,y_train_poly)))

    print(' R2 for train data is',(r2_score(y_train,y_train_poly))*100)

    

    #Evaluation on test data

    print('RMSE for test data is',np.sqrt(mean_squared_error(y_test,y_test_poly)))

    print(' R2 for test data is',(r2_score(y_test,y_test_poly))*100)

create_polynomial(5)