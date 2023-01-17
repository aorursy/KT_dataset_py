import pandas as pd

# import numpy as np

# import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/insurance.csv")

print(df.info())

print(df.dtypes)
df.describe()
Y = df.charges

X = df.drop(columns='charges')

print(X.head())

Y.head()
#One Hot Encoding. get_dummies first converts into labels and then performs one hot encoding

X_dummy = pd.get_dummies(X.loc[:,['sex','smoker','region']])

X_dummy.head()
X.drop(labels=['sex','smoker','region'],axis=1,inplace=True)

X = pd.concat([X,X_dummy],axis=1)

X.head()
#Dropping dependent columns which resulted from one hot encoding of pd.get_dummies() to 

# avoid multicolinearity while still preserving information. e.g. smoker_yes is enough to provide information of 

# an observation.

X.drop(labels=['smoker_yes','sex_female','region_northwest'],axis = 1, inplace=True)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
lm = LinearRegression(normalize=True).fit(X_train,Y_train,)

pred = lm.predict(X_test)

pred_train = lm.predict(X_train)

print("R square on training: ",lm.score(X_train,Y_train))

print("R square on test: ",lm.score(X_test,Y_test))

print("RMSE on Training: ",sqrt(mean_squared_error(Y_train,pred_train)))

print("RMSE on Testing: ",sqrt(mean_squared_error(Y_test,pred)))
