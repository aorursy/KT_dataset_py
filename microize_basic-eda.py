import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
train_shape=train.shape
test_shape=test.shape
print('Train shape:',train_shape)
print('Test shape:',test_shape)
pd.set_option('display.max_columns',None)
display(train.head(3))
train_col=train.columns.to_list()
print('Number of columns:',len(train_col))
print(train.info())
y=train['SalePrice']
X=train.drop(columns=['SalePrice'],axis=1)
int_col=[]
obj_col=[]
for column in X.columns:
    if X[column].dtype=='int64' or X[column].dtype=='float64':
        int_col.append(column)
    else:
        obj_col.append(column)
print("Column with Numerical values:",int_col,'\n\n')
print("Column with Categorical Values:",obj_col)
pd.set_option('display.max_columns',None)
null=X.isnull().sum().to_list()
for x in zip(null,X.columns):
    if x[0]!=0:
        if x[1] in int_col:
            print(x)
X['LotFrontage'].isnull()
X.LotFrontage.fillna(0,inplace=True)
X.MasVnrArea.fillna(0,inplace=True)
X.GarageYrBlt.fillna(0,inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X[int_col],y,test_size=0.20,random_state=3)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(test)
y_predict1=lr.predict(X_test)
mse=mean_squared_error(y_test,y_predict)
mse