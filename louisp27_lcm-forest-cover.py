# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn import preprocessing, pipeline, ensemble

from sklearn import model_selection

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import mean_squared_log_error

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

import xgboost

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()

XGB = xgboost.XGBClassifier()

LR = LogisticRegression(multi_class="multinomial")

GBC = GradientBoostingClassifier()





ohe = preprocessing.OneHotEncoder(categories="auto")

OVR = OneVsRestClassifier(GBC)



X = train.drop(["Id", "Cover_Type"], axis=1)

y = train["Cover_Type"]

model_selection.cross_val_score(GBC, X, y, cv=5, scoring="accuracy")

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size =0.2,random_state =8)

GBC.fit(X_train, y_train)

y_pred = GBC.predict(test.drop(["Id"], axis=1))

#print("The RMSE is: {}".format(sqrt(mean_squared_log_error(y_val, y_pred ))))

submission = pd.DataFrame(

{

    "Id": test["Id"],

    "Cover_Type": y_pred

})

submission.to_csv("submission.csv", index=False)