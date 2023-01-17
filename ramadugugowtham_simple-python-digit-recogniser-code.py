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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
ntrain=pd.read_csv('../input/train.csv')
ntest=pd.read_csv('../input/test.csv')    
ntrain.shape
ntest.shape
ntrain.head()
# Create target vector
y = ntrain.label
### Create features vector
x = ntrain.drop('label', axis=1)
x.shape
# Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# Create random forest classifer object that uses entropy
n_estimators=100
clf = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1, n_estimators=n_estimators)
# Train model
model = clf.fit(x, y)
model.score(x,y)
# Predict observation's class    
y_pred=model.predict(x)
y_pred
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)
y_rf=model.predict(ntest)
y_rf
solution = pd.DataFrame({"ImageId": ntest.index+1, "Label":y_rf})
solution.to_csv("new_digit_rf_submission.csv", index = False)
pd.value_counts(y).plot(kind='bar', legend =True, table= True, figsize=(10,10))
pd.value_counts(y).plot(kind='pie', legend =True, table= True, figsize=(10,10))
pd.value_counts(y).plot(kind='density', legend =True, figsize=(10,10))
#### With Standardisation
# Load libraries
from sklearn import preprocessing
# Create scaler
scaler = preprocessing.StandardScaler()
# Transform the feature
x_new = scaler.fit_transform(x)
# Print mean and standard deviation
print('Mean:', round(x_new[:,0].mean()))
print('Standard deviation:', x_new[:,0].std())
# Train model
model = clf.fit(x_new, y)
clf.score(x_new, y)
test_new=scaler.fit_transform(ntest)
y_rf1=clf.predict(test_new)
y_rf1
confusion_matrix(y_rf, y_rf1)
solution = pd.DataFrame({"ImageId": ntest.index+1, "Label":y_rf1})
solution.to_csv("new_rfscaled_submission.csv", index = False)
from sklearn.ensemble import GradientBoostingClassifier
n_estimators=100
clf = GradientBoostingClassifier(n_estimators=n_estimator)
clf.fit(x,y)
clf.score(x,y)
y_gbm=clf.predict(ntest)
pd.crosstab(y_rf,y_gbm)
solution = pd.DataFrame({"ImageId": ntest.index+1, "Label":y_gbm})
solution.to_csv("new_gbm_submission.csv", index = False)