import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score

from sklearn.metrics import classification_report , confusion_matrix





from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import os

import warnings

warnings.filterwarnings('ignore')



print(os.listdir("../input/"))
train = pd.read_csv('../input/train.csv' , index_col= 'id')

test = pd.read_csv('../input/test.csv' , index_col= 'id')

train.head()
label = train[['target']]

train = train.drop('target' , axis= 1)
for col in train.columns:

    if len(train[col].unique()) < 1000:

        print(col)
train = pd.get_dummies(data = train ,columns=['wheezy-copper-turtle-magic'] , drop_first = True)

test = pd.get_dummies(data = test ,columns=['wheezy-copper-turtle-magic'] , drop_first = True)
train.shape[-1] == test.shape[-1]
scaler = StandardScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)
clf = LogisticRegression()

cross_val_score(clf , train , label , cv = 3)
clf.fit(train , label)
pre = clf.predict_proba(test)
submit = pd.read_csv('../input/sample_submission.csv')

submit['target'] = pre[:,1]
submit.head()
submit.to_csv('submit.csv' , index = False)