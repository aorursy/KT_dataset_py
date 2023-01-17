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
## Uploading the data



titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')



titatnic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
##take a peek at the top few rows of the training set:



titanic_train.head()
## to see if there's any data missing



titanic_train.info()
## .describe() function to further explore the data



titanic_train.describe()
## Import BaseEstimator and TransformerMixin from sklearn.base



from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
## For the numerical data; DataFrameSelector to select all the numerical columns 

## SimpleImputer to fill the missing values with the median



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

    ])
## fit the training data in this pipeline



num_pipeline.fit_transform(titanic_train)
## We will also need an imputer for the categorical columns 

## the regular SimpleImputer does not work on those



class MostFrequentImputer (BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
## Import OneHotEncoder from sklearn.preprocessing 

## This creates a binary column for each category and returns a sparse matrix or dense array



from sklearn.preprocessing import OneHotEncoder
## Pipeline for catagorical attributes



cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])
## fit the training data into this pipeline



cat_pipeline.fit_transform(titanic_train)
## let's join the numerical and categorical pipelines with FeatureUnion



from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])
## Fit the training data in this pipeline



X_train = preprocess_pipeline.fit_transform(titanic_train)



## Save the labels in y_train



y_train = titanic_train["Survived"]
## Train a RandomForest Classifier



from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf = 10, random_state=42)

forest_clf.fit(X_train, y_train)
## Evaluate the model using cross_val_score



from sklearn.model_selection import cross_val_score



forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_test.head()
X_test_prepared = preprocess_pipeline.transform(titanic_test)
final_predictions = forest_clf.predict(X_test_prepared)
final_predictions
output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': final_predictions})

output.to_csv('my_submission.csv', index = False) 