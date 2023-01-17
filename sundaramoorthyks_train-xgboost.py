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
from numpy import loadtxt

from xgboost import XGBClassifier

from matplotlib import pyplot
train_data=pd.read_csv("/kaggle/input/traindatalevel0/Train_data_for_level_1.csv")

test_data=pd.read_csv("/kaggle/input/traindatalevel0/Test_data_for_level_1.csv")
train_data.columns
for col in ['Unnamed: 0','col_1']:

    del train_data[col]

train_y=train_data["result_train"]

#test_data.head()

for col in ['Unnamed: 0','col_1']:

    del test_data[col]

test_y=test_data["result_test"]
del train_data["result_train"]
del train_data["TransactionID"]
del test_data["result_test"]

del test_data["TransactionID"]


test_data.columns
train_y.head()
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0.2,

              learning_rate=0.01, max_delta_step=0, max_depth=7,

              min_child_weight=2, missing=None, n_estimators=1000, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=1, subsample=1, verbosity=1,tree_method='gpu_hist')

model.fit(train_data, train_y)

# feature importance

#print(model.feature_importances_)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()
from xgboost import plot_importance

plot_importance(model)

pyplot.show()
print(model)
test_data.columns=train_data.columns

print(test_data.columns)
train_data.columns
y_pred = model.predict(test_data)

predictions = [round(value) for value in y_pred]
from sklearn.metrics import confusion_matrix, classification_report

c_m=confusion_matrix(test_y,predictions)
c_m
from sklearn.metrics import classification_report

print(classification_report(test_y,predictions))
from sklearn.metrics import accuracy_score

accuracy_score(test_y,predictions )