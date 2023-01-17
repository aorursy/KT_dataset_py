
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

##Load Data

df=pd.read_csv('../input/profit-prediction/datasets_856000_1459830_online.csv')
df
##Shape The Dataset

df.shape
##Information of Data Set
df.info()
## Whole Scenarion of Dataset

df.describe().round(2)
##Check if there is any null value or not

df.isnull().sum()
## No Null Value
## if there then we will be followed this method
##missing=df.profit.mean()
##df.profit=df.profit.fillna(missing)
x=df.drop('Profit',axis=1)
x
y=df['Profit']
y
## Modified Area Column into Categorical Data bcz its String

x=pd.get_dummies(x,columns=['Area'])
x.head(5)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.25,random_state=0)
Lg=LinearRegression()
Lg
#Train The Model
Lg.fit(xtrain,ytrain)
ytest ##The Test size value (25%)
##Predict the Value of Y (profit) According to X features

pred=Lg.predict(xtest)
pred.round(2)
Lg.score(xtest,ytest) 
from sklearn.metrics import r2_score
Accuracy=r2_score(ytest,pred)
Accuracy
x.head(5)
