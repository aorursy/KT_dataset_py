# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.describe())
print(train.shape)
print(train.info())
print(train.columns)
drop_column=['Id']

train_col_len=train.shape[0]
test_col_len=test.shape[0]
for title in train.columns.drop(['SalePrice']):
    if ((train[title].isnull().sum()/train_col_len)>=0.15) or ((test[title].isnull().sum()/test_col_len)>=0.15):
        drop_column.append(title)
train.drop(drop_column,axis=1,inplace=True)
test.drop(drop_column,axis=1,inplace=True)
na_values=[]
for title in train.columns:
    if train[title].isnull().sum()!=0:
        print(title,train[title].isnull().sum(),train[title].dtypes)
        na_values.append(title)
for title in train.select_dtypes(exclude='object').drop(['SalePrice'],axis=1):
    train[title].fillna(train[title].mean(),inplace=True)
    test[title].fillna(test[title].mean(),inplace=True)  
for title in train.select_dtypes(include='object'):
    train[title].fillna(train[title].mode()[0],inplace=True)
    test[title].fillna(test[title].mode()[0],inplace=True)
print(train.info())
numerical=train.select_dtypes(exclude='object').columns.drop('YrSold')
categorical=train.select_dtypes(include='object').columns
from sklearn.preprocessing import LabelEncoder
lab_en=LabelEncoder()
for title in train.select_dtypes(include='object').columns:
    train[title]=lab_en.fit_transform(train[title])
    test[title]=lab_en.fit_transform(test[title])
train['YrSold']=lab_en.fit_transform(train['YrSold'])
test['YrSold']=lab_en.fit_transform(test['YrSold'])
train.info()
plt.subplots(figsize=(20, 20))
sns.heatmap(train.corr(),annot=True)
plt.plot()
Corr=train.corr()
plt.subplots(figsize=(15,15))
sns.heatmap(train[Corr.index[abs(Corr['SalePrice']>0.3)]].corr(),annot=True)
plt.show()
Corr=train.corr()['SalePrice'].abs().sort_values(ascending=False).head(20)
print(Corr)
x=train[Corr.index[abs(Corr['SalePrice']>0.5)][0]].drop(['SalePrice'],axis=1)
y=train['SalePrice']
test=test[x.columns]
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
models=[('Linear_Regression',LinearRegression()),('random_forest',RandomForestRegressor()),
       ('AdaBoost_Regressor',AdaBoostRegressor())]
from sklearn.model_selection import cross_val_score
def regression(name,model):
    score=cross_val_score(model,x,y,cv=5)
    print(name,' : ',score.mean())
for name,model in models:
    regression(name,model)
model=RandomForestRegressor()

model.fit(x,y)
y_pred=model.predict(test)
y_pred=pd.DataFrame(data=y_pred,index=range(1461,2920),columns=['SalePrice'])
y_pred.index.name='Id'
y_pred.to_csv("submission.csv")