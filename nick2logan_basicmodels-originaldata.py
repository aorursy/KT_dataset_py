# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data = train_data.drop(['ID_code'],axis=1)

train_data.columns
# define training and testing sets



Y_train = train_data["target"]

X_train = train_data.drop("target",axis=1)



X_test  = test_data.drop("ID_code",axis=1).copy()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.3, random_state=0)
xgb = XGBClassifier()

xgb.fit(x_train, y_train)
xgb_score = xgb.score(x_test, y_test)



print(xgb_score)
y_pred = xgb.predict(X_test)
submission = pd.DataFrame({

        "ID_code": test_data["ID_code"],

        "target": y_pred

    })

submission.to_csv('submission_logistic.csv', index=False)