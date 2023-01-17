# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sample=pd.read_csv('../input/sample_submission.csv')
train.columns
train.describe()
train.info()
cols= [x for x in train.columns]
cat_var=[]

num_var=[]

for col in cols :

    if train[col].dtype=='object':

        cat_var.append(col)

    if train[col].dtype==('float64') or train[col].dtype==('int64'):

        num_var.append(col)

print("cat_var "+str(len(cat_var))+" num_var "+str(len(num_var)))
df=train.copy()
for f in df.columns:

    if df[f].dtype == 'object':

        label = LabelEncoder()

        label.fit(list(df[f].values))

        df[f] = label.transform(list(df[f].values))

df.head()
X=df.drop(["SalePrice"],axis=1)

Y=df.SalePrice

X.shape
train_x, test_x, train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=2)
model=xgb.XGBRegressor()
model.fit(train_x,train_y)
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())
output=model.predict(test_x)
rmse(output,test_y)
model.fit(X,Y)
#test data

for f in test.columns:

    if test[f].dtype == 'object':

        label = LabelEncoder()

        label.fit(list(test[f].values))

        test[f] = label.transform(list(test[f].values))

test.head()
output=model.predict(test)
sample.columns
sample['SalePrice']=output

sample.to_csv('sub_sample_mod.csv')
sub=pd.DataFrame()

sub['Id']=test['Id']

sub['SalePrice']=output

sub.shape

sub.to_csv('submission1.csv',index=False)