import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
# What does the data look like#

train_data.head()
# More information

train_data.info()
train_data.describe()
train_data["Survived"].value_counts()
# Checking Categorical attributes

train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



# Numerical Pipeline

num_pipeline = Pipeline([

    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),

    ("imputer", SimpleImputer(strategy="median")),

])
num_pipeline.fit_transform(train_data)
# Categorical Pipeline

class MostFrequent(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder
cat_pipeline = Pipeline([

    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

    ("imputer", MostFrequent()),

    ("cat_encoder", OneHotEncoder(sparse=False)),

])
cat_pipeline.fit_transform(train_data)
# Combine Numerical and Categorical pipelines

from sklearn.compose import ColumnTransformer



preprocess_pipeline = ColumnTransformer([

    ("num_pipeline", num_pipeline, ["Age", "SibSp", "Parch", "Fare"]),

    ("cat_pipeline", cat_pipeline, ["Pclass", "Sex", "Embarked"]),

])
X_train = preprocess_pipeline.fit_transform(train_data.drop(["Survived"], axis=1))

X_train
y_train = train_data["Survived"]
X_test = preprocess_pipeline.transform(test_data)
# 1. SVM

from sklearn.svm import SVC



svm_clf = SVC(gamma="auto")

svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
# What is the cross val score for this one?

from sklearn.model_selection import cross_val_score



svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10, n_jobs=-1)

svm_scores.mean()
# 2. Random Forest

from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10, n_jobs=-1)

forest_scores.mean()
forest_clf.fit(X_train, y_train)

y_final_preds = forest_clf.predict(X_test)
# Submission

PassengerId = test_data['PassengerId']

Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': y_final_preds })

Submission.to_csv("submission.csv", index=False)