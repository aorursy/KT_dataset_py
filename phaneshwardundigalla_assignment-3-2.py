#importing all libraries 

import numpy as np

import pandas as pd 

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,r2_score

df = pd.read_csv("../input/hiring.csv")

df

df["experience"][0]=0

df["experience"][1]=0

df["experience"][2]=5

df["experience"][3]=2

df["experience"][4]=7

df["experience"][5]=3

df["experience"][6]=10

df["experience"][7]=11

df["experience"]=pd.to_numeric(df["experience"])

df["test_score(out of 10)"][6]=0

df
x = df.iloc[:, 1:3].values

y = df.iloc[:, -1].values

y

#creating with predifined 

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5,random_state=16)

print(xtrain[0][0])
#Creating with our own functions

def Train_Test_Split(x,y,test_size=0.5,random_state=None):

    n=len(y)

    if len(x)==len(y):

        if random_state:

            np.random.seed(random_state)

        shuffle_index=np.random.permutation(n)

        x=x[shuffle_index]

        y=y[shuffle_index]

        test_data=round(n*test_size)

        xtrain,xtest=x[test_data:],x[:test_data]

        ytrain,ytest=y[test_data:],y[:test_data]

        return xtrain,xtest,ytrain,ytest

    else:

        print("Data should be in same size.")

        
xtrain,xtest,ytrain,ytest=Train_Test_Split(x,y,test_size=0.3,random_state=12)

model=LinearRegression()

model.fit(xtrain,ytrain)

m=model.coef_

c=model.intercept_

x=[2,9,6]

y_pred=sum(m*x)+c

y_pred
x=[12,10,10]

y_pred=sum(m*x)+c

y_pred