# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.metrics import roc_auc_score

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
X = train_df.drop(['ID_code','target'], axis=1) # Features

y = train_df.target.values # Target variable



X_test_pred = test_df.drop(['ID_code'], axis=1)
#Se divide el dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)  #tamaño cambiarlo a 0.25
modelo = xgb.XGBClassifier(max_depth=8, min_child_weight=1,  n_estimators=200, n_jobs=2, verbose=1, learning_rate=0.25)

modelo.fit(X_train,y_train)
# make predictions for test data

y_pred = modelo.predict(X_test)

predictions = [round(value) for value in y_pred]
from sklearn.metrics import accuracy_score

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
roc_auc_score(y_test, y_pred)
y_pred_test = modelo.predict(X_test_pred)
#creacion del archivo

submission = pd.DataFrame({'ID_code':test_df.ID_code,'target':y_pred_test})

submission.to_csv('submission.csv', index=False)