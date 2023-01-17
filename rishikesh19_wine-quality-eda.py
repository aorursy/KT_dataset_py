import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
import math
df=pd.read_csv('winequality-red.csv')
df.head()
df.describe()
df.shape
df.isnull().sum()
df.corr()
X = df.iloc[:, 0:10]  
y = df['quality'].values.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)
regressor=LinearRegression()
regressor.fit(X_train,y_train)

print(regressor.intercept_)
print(regressor.coef_)
y_pred=regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

rmse = math.sqrt(mse)
print(mse)

