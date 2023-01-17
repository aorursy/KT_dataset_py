import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.info()
train_data.describe()
train_data['Survived'].value_counts()
from sklearn.base import BaseEstimator, TransformerMixin



class AttributeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes_names):

        self.attributes_names = attributes_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attributes_names]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



numerical_pipeline = Pipeline([

    ('select_numeric', AttributeSelector(['Age', 'SibSp', 'Parch'])),

    ('imputer', SimpleImputer(strategy='median')) # Replacing with median values

])
# Using the `numerical_pipeline`

numerical_pipeline.fit_transform(train_data)
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[att].value_counts().index[0] for att in X],

                                       index=X.columns)

        return self

    

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder
categorical_pipeline = Pipeline([

    ('select_cat', AttributeSelector(['Pclass', 'Sex', 'Embarked'])),

    ('imputer', CategoricalImputer()),

    ('cat_encoder', OneHotEncoder(sparse=False)),

])
categorical_pipeline.fit_transform(train_data)
from sklearn.pipeline import FeatureUnion



preprocess_pipeline = FeatureUnion(transformer_list=[

    ('numerical_pipeline', numerical_pipeline),

    ('categorical_pipeline', categorical_pipeline),

])
X_train = preprocess_pipeline.fit_transform(train_data)

y_train = train_data['Survived']
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict



rnd_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

rnd_forest_clf.fit(X_train, y_train)

rnd_forest_scores = cross_val_score(rnd_forest_clf, X_train, y_train, cv=10)

rnd_forest_scores.mean()
X_test = preprocess_pipeline.transform(test_data)

y_pred = rnd_forest_clf.predict(X_test)

y_pred
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)