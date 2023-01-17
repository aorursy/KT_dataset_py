# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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

# supervised, binary classification problem

# measure with "accuracy"
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.info() # age,Cabin is missing
train_data.describe()
train_data["Survived"].value_counts()
_ = train_data.hist(bins=50, figsize=(20, 15))
train_data.shape, test_data.shape
train_data_copy = train_data.copy()
# fill out missing value

Age_median = train_data_copy["Age"].median()

train_data_copy["Age"].fillna(Age_median, inplace=True)
# see how values are correlated to Survived

corr_matrix = train_data_copy.corr()

corr_matrix["Survived"].sort_values(ascending=False)
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



num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

    ])
# Inspired from stackoverflow.com/questions/25239958

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
# for better understanding of above code, tryout

data = [

    ['a', 1, 2],

    ['b', 1, 1],

    ['b', 2, 2],

    [np.nan, np.nan, np.nan]

]



X = pd.DataFrame(data)

X[0].value_counts().index[0]
from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])
from sklearn.compose import ColumnTransformer

train_data_copy = train_data.copy()

full_pipeline = ColumnTransformer([

        ("num", num_pipeline, ["Age", "SibSp", "Parch", "Fare"]),

        ("cat", cat_pipeline, ["Pclass", "Sex", "Embarked"])

    ])

new_attr1 = train_data_copy["Pclass"].value_counts(sort=False)

new_attr2 = train_data_copy["Sex"].value_counts(sort=False)

new_attr3 = train_data_copy["Embarked"].value_counts(sort=False)

train_data_prepared = full_pipeline.fit_transform(train_data_copy)

pd.DataFrame(train_data_prepared, columns= ["Age", "SibSp", "Parch", "Fare"] + list(new_attr1.index) + list(new_attr2.index) + list(new_attr3.index),

            index = train_data_copy.index)
# compare with the above data. it matches

train_data_copy.head()
y_train = train_data_copy["Survived"].copy()
# model 1

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



param_grid = {

        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

    }



svm_clf = SVC(random_state=42)

rnd_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy',

                          verbose=2, n_jobs=-1,

                          return_train_score=True)

rnd_search.fit(train_data_prepared, y_train)
rnd_search.cv_results_["mean_test_score"]

# not good
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier



param_grid = [

    {'max_iter': [500, 1000, 3000], 'tol': [1e-3, 1e-4, 1e-5]}

  ]





sgd_clf = SGDClassifier(n_jobs= -1, random_state=42)

grid_search = GridSearchCV(sgd_clf, param_grid, cv=5,

                           scoring='accuracy',

                           return_train_score=True)

grid_search.fit(train_data_prepared, y_train)
grid_search.cv_results_["mean_test_score"]

# result not good
# model_3

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



param_grid = [

    {'n_estimators': [150, 200, 250], 'max_depth': [None, 3, 4, 5], 'oob_score':[True, False]}

  ]

rnd_clf = RandomForestClassifier(n_jobs = -1, random_state=42)

grid_search = GridSearchCV(rnd_clf, param_grid, cv=5,

                           scoring='accuracy', n_jobs=-1,

                           return_train_score=True)

grid_search.fit(train_data_prepared, y_train)
grid_search.cv_results_["mean_test_score"]

# better
final_model = grid_search.best_estimator_

final_model

# best model found with max_depth 4, oob_score = True, n_estimators = 200
test_data_prepared = full_pipeline.fit_transform(test_data)

predictions = final_model.predict(test_data_prepared)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")