# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_NIR5Yl1.csv')

train.head()
train['Tag'].nunique()
train.info()
df = pd.get_dummies(train,drop_first=True)

df.head()
df.isnull().sum()
from sklearn.model_selection import StratifiedKFold,train_test_split

X_train,X_val,y_train,y_val = train_test_split(df.drop(['Upvotes'],axis=1),df['Upvotes'],test_size=0.25,random_state = 1994)
df.shape
print('X_Train',X_train.shape)

print('X_val',X_val.shape)



print('y_train',y_train.shape)

print('y_val',y_val.shape)
X_val.head()
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train,y_train)

p = rf.predict(X_val)
from sklearn.metrics import mean_squared_error,accuracy_score

print(np.sqrt(mean_squared_error(y_val,p)))
test = pd.read_csv('../input/test_8i3B3FC.csv')

test.head()
dftest = pd.get_dummies(test,drop_first=True)

dftest.shape
rf.fit(df.drop(['Upvotes'],axis=1),df['Upvotes'])

y_pred = rf.predict(dftest)
test['Upvotes'] = y_pred
test.head()
test[['ID','Upvotes']].to_csv('basicrf.csv',index=False)