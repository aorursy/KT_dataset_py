import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from math import sqrt

import os

%matplotlib inline

df=pd.read_csv("../input/Admission_Predict.csv")
df.head()
df.isnull().any()
df.describe()
df.info()
for i in df.columns:

    x=len(df[i].unique())

    if i not in ['Serial No.','Chance of Admit ']:

        if int(x)<10 or df[i].dtype=='object':

            sns.boxplot(x=i,y='Chance of Admit ' ,data=df)

            plt.show()
def pre(X,Y):

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y)



    LR=LinearRegression()

    LR.fit(X_train,Y_train)



    my_predict=LR.predict(X_test)



    error=sqrt(metrics.mean_squared_error(Y_test,my_predict))

    

    return error

    
X=df.iloc[:,1:-1] #selecting all columns except serial no. and chance of admit

Y=df.iloc[:,-1]   #selecting column chance of admit



error=pre(X,Y)

print("Root Mean Square Error: {}".format(error))
X2=df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'CGPA']] #selecting all columns except serial no. and chance of admit

Y2=df.iloc[:,-1]   #selecting column chance of admit



error=pre(X2,Y2)

print("Root Mean Square Error: {}".format(error))