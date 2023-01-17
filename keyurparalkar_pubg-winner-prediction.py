# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install fastai==0.7.0
from fastai import *

from fastai.structured import *
from sklearn.model_selection import train_test_split

from sklearn.ensemble import *
PATH = '../input/pubg-finish-placement-prediction/'
!ls {PATH}
train_data = pd.read_csv(PATH+'train_V2.csv')
pd.set_option('display.max_columns',500)
train_data.head()


all_objects = [i for i,dtype in enumerate(list(train_data.dtypes)) if(dtype=='object')]

train_data[train_data.columns[all_objects]]
train_cats(train_data)
train_data.Id.cat.categories
train_data.isnull().sum().sort_index()
train_data.shape
train_data = train_data.drop(train_data[train_data.winPlacePerc.isna()].index)
# #Saving the raw dataset:

# os.makedirs('tmp',exist_ok=True)

# train_data.to_feather('tmp/train_raw.csv')
#Loadint the preprocessed dataset:

# df_raw = pd.read_feather('tmp/train_raw.csv')
train_raw, y, nas = proc_df(train_data,'winPlacePerc')
X_train, X_valid, y_train, y_valid = train_test_split(train_raw, y)
print(f"Training dataset size = {X_train.shape},{y_train.shape}")

print(f"Validation dataset size = {X_valid.shape},{y_valid.shape}")
from sklearn.metrics import *
#For printing mean absolute error, r^2

def print_score(m):

    print([mean_absolute_error(m.predict(X_train),y_train),mean_absolute_error(m.predict(X_valid),y_valid),\

           m.score(X_train,y_train),m.score(X_valid,y_valid)])
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)

%time m.fit(X_train,y_train)

print_score(m)
#After setting rf samples and then running randomForestRegressor

set_rf_samples(1000000)
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)

%time m.fit(X_train,y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)

%time m.fit(X_train,y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=80,min_samples_leaf=5, n_jobs=-1)

%time m.fit(X_train,y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=100,min_samples_leaf=5, n_jobs=-1)

%time m.fit(X_train,y_train)

print_score(m)
test_data = pd.read_csv(PATH+'test_V2.csv')
test_data.shape
test_data.columns, train_data.columns
#Applying preprocessing to test_data

train_cats(test_data)

test,x,_ = proc_df(test_data)
test_preds = m.predict(test)
#viewing sample submission:

pd.read_csv(PATH+'sample_submission_V2.csv')
# test_data.Id

test_preds.shape
PATH_sub = '/kaggle/working/'     #Save your submission file at this location since we can read and write from this location.

test_dict = {'Id':test_data.Id,'winPlacePerc':test_preds}

test_sub = pd.DataFrame(test_dict,columns=["Id","winPlacePerc"])

test_sub.to_csv('submission.csv',index=False)
# from IPython.display import FileLink

# FileLink('submission.csv')