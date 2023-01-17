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
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
train_df['fuel-type'] = train_df['fuel-type'].map({'diesel':1,'gas':2})

train_df['aspiration'] = train_df['aspiration'].map({'std':1,'turbo':2})

train_df['num-of-doors'] = train_df['num-of-doors'].map({'four':1,'two':2})

train_df['body-style'] = train_df['body-style'].map({'fardtop':1,'wagon':2,'sedan':3,'hatchback':4,'convertible':5})

train_df['drive-wheels'] = train_df['drive-wheels'].map({'4wd':1,'fwd':2,'rwd':3})

train_df['engine-type'] = train_df['engine-type'].map({'dohc':1,'dohcv':2,'l':3,'ohc':4,'ohcf':5,'ofcv':6,'rotor':7})

train_df['num-of-cylinders'] = train_df['num-of-cylinders'].map({'eight':1,'five':2,'four':3,'six':4,'three':5,'twelve':6,'two':7})

train_df['fuel-system'] = train_df['fuel-system'].map({'1bbl':1,'2bbl':2,'4bbl':3,'idi':4,'mfi':5,'mpfi':6,'spdi':7,'spfi':8})
test_df['fuel-type'] = test_df['fuel-type'].map({'diesel':1,'gas':2})

test_df['aspiration'] = test_df['aspiration'].map({'std':1,'turbo':2})

test_df['num-of-doors'] = test_df['num-of-doors'].map({'four':1,'two':2})

test_df['body-style'] = test_df['body-style'].map({'fardtop':1,'wagon':2,'sedan':3,'hatchback':4,'convertible':5})

test_df['drive-wheels'] = test_df['drive-wheels'].map({'4wd':1,'fwd':2,'rwd':3})

test_df['engine-type'] = test_df['engine-type'].map({'dohc':1,'dohcv':2,'l':3,'ohc':4,'ohcf':5,'ofcv':6,'rotor':7})

test_df['num-of-cylinders'] = test_df['num-of-cylinders'].map({'eight':1,'five':2,'four':3,'six':4,'three':5,'twelve':6,'two':7})

test_df['fuel-system'] = test_df['fuel-system'].map({'1bbl':1,'2bbl':2,'4bbl':3,'idi':4,'mfi':5,'mpfi':6,'spdi':7,'spfi':8})
train_df.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')



train_df['num-of-doors'] = imputer.fit_transform(train_df['num-of-doors'].values.reshape(-1, 1))

train_df['body-style'] = imputer.fit_transform(train_df['body-style'].values.reshape(-1, 1))

train_df['engine-type'] = imputer.fit_transform(train_df['engine-type'].values.reshape(-1, 1))
test_df.isnull().sum()
test_df['body-style'] = imputer.fit_transform(test_df['body-style'].values.reshape(-1, 1))

test_df['engine-type'] = imputer.fit_transform(test_df['engine-type'].values.reshape(-1, 1))
#numeric_columns = ['fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-type','num-of-cylinders','fuel-system','wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'compression-ratio', 'city-mpg', 'highway-mpg']
numeric_columns = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'compression-ratio', 'city-mpg', 'highway-mpg']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_df[numeric_columns], train_df['symboling'], test_size=0.2, random_state=0)
X_train = X_train.to_numpy()

X_valid = X_valid.to_numpy()

y_train = y_train.to_numpy()

y_valid = y_valid.to_numpy()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
from collections import Counter

Counter(y_resampled)
import xgboost as xgb

from sklearn import datasets

from sklearn import model_selection

from sklearn.metrics import confusion_matrix, mean_squared_error

import sklearn.preprocessing as sp

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math
clf = xgb.XGBRegressor()

clf.fit(X_resampled,y_resampled)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor() #ランダムフォレスト

rfr.fit(X_resampled, y_resampled)
#ランダムフォレスト

y_train_pred = rfr.predict(X_valid)

np.sqrt(mean_squared_error(y_valid, y_train_pred))
#XGBoost

y_train_pred = clf.predict(X_valid)

np.sqrt(mean_squared_error(y_valid, y_train_pred))
X_test = test_df[numeric_columns].to_numpy()

p_test = clf.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')