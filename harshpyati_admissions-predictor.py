import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
df = pd.read_csv('../input/Admission_Predict.csv')
df.info()
df.head()
sns.heatmap(df.corr(),cmap ='coolwarm')
df.shape
#It has around 400 rows, so a 380 + 20 split would be good i suppose
df.columns
X = df.drop('Chance of Admit ',axis = 1)
y = df['Chance of Admit ']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.05)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error
import math
print(math.sqrt(mean_squared_error(y_test,pred)))
## The rmse value is good enough, but let's try using some other models
from sklearn.tree import DecisionTreeRegressor
dcr = DecisionTreeRegressor()
dcr.fit(X_train,y_train)
predDecTree = dcr.predict(X_test)
print(math.sqrt(mean_squared_error(y_test,predDecTree)))
## The rmse value increased by using Decision Tree regressor. Maybe the linear model is better. 
from sklearn.ensemble import RandomForestRegressor
rfR = RandomForestRegressor(n_estimators=10)
rfR.fit(X_train,y_train)
predRF = rfR.predict(X_test)
print(math.sqrt(mean_squared_error(y_test,predRF)))
from sklearn.svm import SVR
svR = SVR(kernel = 'rbf')
svR.fit(X_train,y_train)
predSVR = svR.predict(X_test)
print(math.sqrt(mean_squared_error(y_test,predSVR)))



