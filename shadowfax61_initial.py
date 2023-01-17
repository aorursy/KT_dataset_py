# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test= pd.read_csv("../input/test.csv")

train= pd.read_csv("../input/train.csv")
print(train.columns)
a= train["datetime"][2]

a
import datetime
b= datetime.datetime.strptime(a, '%Y-%m-%d %X')
int(b.hour)
train["hour"]= train["datetime"].apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d %X').hour))
test["hour"]= test["datetime"].apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d %X').hour))
train_features =train[['hour', 'season', 'holiday', 'workingday', 'weather', 'temp',

       'atemp', 'humidity', 'windspeed']]
train_features

train_target=train[['casual', 'registered', 'count']]
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

param = {'objective':'reg:logistic' }





# fit model no training data

clf_casual = xgb.XGBClassifier( objective= 'reg:logistic')

clf_registered = xgb.XGBClassifier( objective= 'reg:logistic')
clf_casual.fit(train_features, train_target["casual"])
clf_registered.fit(train_features, train_target["registered"])

test_features =test[['hour', 'season', 'holiday', 'workingday', 'weather', 'temp',

       'atemp', 'humidity', 'windspeed']]
test["count"] = clf_registered.predict(test_features)+clf_casual.predict(test_features)
test.columns
output = test[["datetime","count"]]



output.to_csv("output.csv",index=False)
output
