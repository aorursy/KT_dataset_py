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
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.describe()
train_df.info()
print("Test info")
test_df.info()
# inspect categorial data

train_df["Survived"].value_counts()
train_df["Pclass"].value_counts()
train_df["Sex"].value_counts()
train_df["Embarked"].value_counts()
# build preprocessing pipeline

from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')

num_pipeline = Pipeline([
    ("select_numberic", DataFrameSelector(["Age", "SibSp","Parch","Fare"])),
    ("imputer", Imputer(strategy="median")),
])
num_pipeline.fit_transform(train_df)
# imputer for categorial columns

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
cat_pipeline.fit_transform(train_df)
# Join the numerical and categorical piplelines

from sklearn.pipeline import FeatureUnion
preprocessing_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])
X_train = preprocessing_pipeline.fit_transform(train_df)
X_train
y_train = train_df['Survived']
# data processing completed now check with various classfiers
# let's start with basic SGDClassifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)

X_test = preprocessing_pipeline.transform(test_df)
y_pred = sgd_clf.predict(X_test)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# CROSS VALIDATION

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# kNN Score
round(np.mean(score)*100, 2)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# decision tree Score
round(np.mean(score)*100, 2)
from sklearn.ensemble import RandomForestClassifier
rand_clf = RandomForestClassifier(n_estimators=20, max_leaf_nodes=20, n_jobs=-1)
scoring = 'accuracy'
rand_clf_score = cross_val_score(rand_clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Random Forest Score
round(np.mean(rand_clf_score)*100, 2)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf,  X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Naive Bayes Score
round(np.mean(score)*100, 2)
from sklearn.svm import SVC
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)

# so far random forest gives best result
X_test = preprocessing_pipeline.transform(test_df)
rand_clf.fit(X_train, y_train)
y_pred = rand_clf.predict(X_test)

sub=pd.DataFrame()
sub['PassengerId']=test_df['PassengerId']
sub['Survived']=y_pred
# print(test_df['PassengerId'])
# print(sub.head())
sub.to_csv('./submission.csv', index=False)
