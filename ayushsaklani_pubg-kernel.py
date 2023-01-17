# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
train.head()
test.head()
labels = train['winPlacePerc']
train = train.drop(['winPlacePerc'],axis=1)
train.head()
train.info()
test.isnull
def object_to_num(df,index = 'Id'):
    df.set_index([index],inplace = True)
    for column in df.select_dtypes(include=['object']).columns:
        print(column)
        df[column] = pd.Categorical(df[column])
        df[column] = df[column].cat.codes
    return df
    
train = object_to_num(train)
test= object_to_num(test)
train.head()
test.head()
labels.isnull().sum()
labels.fillna(labels.mean(),inplace = True)
labels.isnull().sum()
xgbreg = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=9, seed=9, missing=None, importance_type='gain')
#scores = cross_val_score(xgbreg, train, y=labels, groups=None, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=15)
xgbreg.fit(train,labels,verbose = 50)

sol = xgbreg.predict(test)
submission = pd.read_csv('../input/sample_submission_V2.csv')
submission.head()
sol = pd.DataFrame({'Id': submission.Id, 'winPlacePerc':sol})
sol.head()
sol.to_csv('submission_pubg.csv',index = False)

