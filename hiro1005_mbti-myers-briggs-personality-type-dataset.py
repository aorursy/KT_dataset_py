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
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/mbti-type/mbti_1.csv', header=0)
train.head(10)
# CountVectorizer(min_dfとストップワード)
train_feature = train['posts'].values.astype('U')
train_target = train['type']

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english').fit(train_feature)
train_feature = vect.transform(train_feature)
print("train_feature with min_df:\n{}".format(repr(train_feature)))

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
# RandomForest==============

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')
# TF-IDF
train = pd.read_csv('/kaggle/input/mbti-type/mbti_1.csv', header=0)
train.head(10)

train_feature = train['posts'].values.astype('U')
train_target = train['type']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
train_feature = tfidf_vect.fit_transform(train_feature)

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
# RandomForest==============

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')