# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")
train.head()
test.head()
print('shape of train=',train.shape)
print('shape of test=',test.shape)
# Approximately same amount od data points in each
#Combining train and test data to reduce steps in categorical transformation
train['train_or_test']='train'
test['train_or_test']='test'
df=pd.concat([train,test])
df.head()
df=df.drop(columns='Id',axis=1)
#Preparing data for XGBoost model
#Getting Dummies for categorical values
df=pd.get_dummies(df)
df.head(2)
#separating train and test data
modified_train=df[df.train_or_test_train==1]
modified_test=df[df.train_or_test_train==0]
print('shape of train data=',modified_train.shape)
print('shape of test data=',modified_test.shape)
# Dropping train_test feature from train and test data
modified_train=modified_train.drop(columns=['train_or_test_train','train_or_test_test'],axis=1)
modified_test=modified_test.drop(columns=['train_or_test_train','train_or_test_test','SalePrice'],axis=1)
#Separating X and y from modified train
x=modified_train.drop(columns='SalePrice',axis=1)
y=modified_train['SalePrice']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1)
from xgboost import XGBRegressor
xgb=XGBRegressor(max_depth=5,n_estimators=100)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print('Mean squared error=',mean_squared_error(y_test,y_pred))
print('Mean absolute error=',mean_absolute_error(y_test,y_pred))
print('r2_score=',r2_score(y_test,y_pred))
#training full data using XGBoost
xgb.fit(x,y)
y_final=xgb.predict(modified_test)
sample_submission['SalePrice']=y_final
sample_submission.head()
sample_submission.to_csv("result.csv",index=False)
