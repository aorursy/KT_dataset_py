# Data file

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost.sklearn import XGBClassifier 

from lightgbm import LGBMClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/spooky-author-identification/train.zip', header=0)

train.head(10)
test = pd.read_csv('/kaggle/input/spooky-author-identification/test.zip', header=0)

test.head(10)
submission = pd.read_csv('/kaggle/input/spooky-author-identification/sample_submission.zip', header=0)

submission.head(10)
train['author'] = train['author'].replace("EAP",0).replace("HPL",1).replace("MWS",2)



train_feature = train['text'].values.astype('U')

train_target = train['author']

test_feature = test['text']



from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=5, stop_words='english').fit(train_feature)

train_feature = vect.transform(train_feature)

test_feature = vect.transform(test_feature)

print("train_feature with min_df:\n{}".format(repr(train_feature)))



X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0)
# RandomForest==============



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

print('='*20)

print('RandomForestClassifier')

print(f'accuracy of train set: {rf.score(X_train, y_train)}')

print(f'accuracy of test set: {rf.score(X_test, y_test)}')



rf_prediction = rf.predict(test_feature)

rf_prediction