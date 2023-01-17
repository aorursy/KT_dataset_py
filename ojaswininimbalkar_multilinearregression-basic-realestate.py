import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split  #to split data

from sklearn.metrics import accuracy_score,r2_score

#for training and testing



from ipywidgets import interact

df=pd.read_csv("../input/multiplelr-realestate/Real estate.csv")    #importing the csv file, locations may vary
df.head() #to show first 5 rows of the dataset
df.shape
x=df.iloc[:,1:4].values # all rows and 3 columns

                        # data in 2D array

x
y=df.iloc[:,-1].values #al rows of last column

y
#split data for training and testing

x.shape
len(train_test_split(x,y,test_size=0.3))  #shows no. of frames returned by this function
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

x_train.shape
model=LinearRegression()

model.fit(x_train,y_train)
m=model.coef_

c=model.intercept_
ytest_predict=model.predict(x_test)

ytest_predict
#Example:



x=[[12,123,7]]  

y_pred=model.predict(x)

print("House price of unit area: ",y_pred)