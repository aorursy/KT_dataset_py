# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

# train_data['family'] = train_data['SibSp'] + train_data['Parch']

# train_data["AgeBucket"] = train_data["Age"] // 15 * 15
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical or categorical columns 

# since Scikit-Learn doesn't handle DataFrames yet

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
from sklearn.pipeline import Pipeline

try:

    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

except ImportError:

    from sklearn.preprocessing import Imputer as SimpleImputer



num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["SibSp", "Parch","Age","Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

    ])
num_pipeline.fit_transform(train_data)

# Inspired from stackoverflow.com/questions/25239958

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])
cat_pipeline.fit_transform(train_data)

from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])
X_train = preprocess_pipeline.fit_transform(train_data)

X_train
y_train = train_data["Survived"]

from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



RFC = RandomForestClassifier()

kfold = StratifiedKFold(n_splits=10)







## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,Y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
RFC_best.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)

y_pred = RFC_best.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv', index=False)