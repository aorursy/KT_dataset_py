import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv('../input/big-mart-sales-forcasting/train.csv')
train.head(10)
train.shape
# check for missing values in data set

train.isnull().sum()
train['Item_Weight'].fillna((train['Item_Weight'].mean()),inplace=True)
train.isnull().sum()
train['Outlet_Size'].fillna((train['Outlet_Size'].mode()[0]),inplace=True)
train.isnull().sum()
X=train[['Outlet_Establishment_Year','Item_MRP']]
X.shape
y=train['Item_Outlet_Sales']
# spliting the data into train and test

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)
from sklearn.linear_model import LinearRegression

lreg=LinearRegression()
lreg.fit(X_train,y_train)
pred=lreg.predict(X_test)
lreg.coef_
mse=mean_squared_error(y_test,pred)
mse
print('the value of mse is',mse)
# calculating coefficients

coeff=pd.DataFrame(X_train.columns)

coeff['coeffucient Estimation']=lreg.coef_
coeff
# from above result we can say that MRP has a high coefficient,meaning items having higher prices have better sales.
X=train[['Outlet_Establishment_Year','Item_MRP','Item_Weight']]
y=train[['Item_Outlet_Sales']]
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)
X_train.shape,y_train.shape
lreg.fit(X_train,y_train)
pred1=lreg.predict(X_test)
lreg.coef_,lreg.intercept_
from sklearn.metrics import r2_score

lreg.score(X_test,y_test)
from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,pred1)
print('the value of mse is',mse)
train.info()
train.drop('Item_Identifier',axis=1,inplace=True)
train.Item_Visibility.value_counts()
train['Item_Visibility']=train['Item_Visibility'].replace(0,np.mean(train['Item_Visibility']))
train.Item_Visibility.value_counts()
train.Outlet_Establishment_Year.value_counts()
train.info()
train.drop('Outlet_Identifier',axis=1,inplace=True)
train1=pd.get_dummies(train,drop_first=True)
train1.head()
X=train1.drop('Item_Outlet_Sales',axis=1)
y=train1['Item_Outlet_Sales']
X.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
X_train.shape,y_train.shape
lreg=LinearRegression()
lreg.fit(X_train,y_train)
pred2=lreg.predict(X_test)
lreg.score(X_test,y_test)
lreg.coef_
# let's check MSE

mse=mean_squared_error(y_test,pred2)
print('the value of mse is',mse)
coeff=pd.DataFrame(X_train.columns)

coeff['Estimated Coefficient']=lreg.coef_
coeff
# Residual Plot

X_plot=plt.scatter(pred,(pred-y_test),c='b')

plt.hlines(y=0,xmin=-1000,xmax=5000)

plt.title('residual plot')

# Residual Plot

X_plot=plt.scatter(pred1,(pred-y_test),c='b')

plt.hlines(y=0,xmin=-1000,xmax=5000)

plt.title('residual plot')

# Residual Plot

X_plot=plt.scatter(pred2,(pred2-y_test),c='b')

plt.hlines(y=0,xmin=-1000,xmax=5000)

plt.title('residual plot')

from pandas import Series

coeff=pd.DataFrame(X_train.columns)

coeff['Estimated Coefficient']=Series(lreg.coef_,X_train.columns).sort_values().plot(kind='bar')