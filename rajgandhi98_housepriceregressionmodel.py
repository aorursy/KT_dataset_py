# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read the data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.info()

test.info()
import seaborn as sns

sns.scatterplot(x=train['MiscFeature'],y=train['SalePrice'])

sns.scatterplot(x=train['Alley'],y=train['SalePrice'])

sns.scatterplot(x=train['Fence'],y=train['SalePrice'])
#combine the data

y_train = train['SalePrice']

train = train.drop('SalePrice',axis = 1)

train = train.drop('PoolQC',axis = 1)

test = test.drop('PoolQC',axis = 1)

train = train.drop('MiscFeature',axis = 1)

test = test.drop('MiscFeature',axis = 1)

train = train.drop('Alley',axis = 1)

test = test.drop('Alley',axis = 1)

train = train.drop('Fence',axis = 1)

test = test.drop('Fence',axis = 1)

X = pd.concat([train,test]).reset_index()

y_train
#data info

X.info()
X.head()  
#cleaning the data

for i in X:

    if X[i].dtype == 'object':

        X[i] = X[i].fillna('missing')

for i in X:

    if X[i].dtype == 'int64' or 'flot64':

        X[i] = X[i].fillna('0')

X.info()
#convert classsified rows into numerical counterparts using encoding

X_dummy_object = pd.get_dummies(X.select_dtypes(include='object'))

X_dummy_num = X.select_dtypes(include='int64' or 'flot64')                                                 
X_dummy_object.info()
X_train_test = pd.concat([X_dummy_object,X_dummy_num],axis=1)

X_train_test.info()
#split the data for training and testing

X_train,X_test = train_test_split(X_train_test,test_size = 0.5,shuffle=False)

X_train.info()

X_test.info()
#we get one extra entry in test dataset so delete that and delete last entry from y_train

del y_train[1459]
#run the regression model

import xgboost

from xgboost import XGBRegressor

final = XGBRegressor(learning_rate = 0.01, n_estimators = 2000, max_depth = 3, subsample = 0.8, random_state = 777)

final.fit(X_train,y_train)
import matplotlib.pyplot as plt

import seaborn as sns

sns.scatterplot(x=X_train['YearBuilt'],y=y_train)
prediction = final.predict(X_test)

prediction.shape
final_submission = pd.DataFrame({'ID':X_test['Id'],'SalePrice':prediction})

final_submission.set_index('ID',inplace=True)

final_submission.head()
final_submission.drop(final_submission.index[0],inplace=True)

final_submission.head()
final_submission.to_csv('final_submission.csv')