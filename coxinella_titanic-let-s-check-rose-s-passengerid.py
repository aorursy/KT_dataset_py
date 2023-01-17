#In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.

#In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
train_data.info()
train_data.describe()
train_data.drop(columns=["PassengerId"]).hist(bins=40, figsize=(15,12))
train_data.Embarked.value_counts()
corr_matrix = train_data.corr()

corr_matrix["Survived"].sort_values(ascending=False)
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
from sklearn.impute import SimpleImputer



num_imputer = SimpleImputer(strategy="median")
class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder(sparse=False)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
from sklearn.pipeline import Pipeline



num_attribs = ["Age", "SibSp", "Parch", "Fare"]



num_pipeline = Pipeline([

    ('select_num', DataFrameSelector(num_attribs)),

    ('imputer', num_imputer),

    ('scaler', scaler)

])
cat_attribs = ["Pclass", "Sex", "Embarked"]



cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(cat_attribs)),

        ("imputer", MostFrequentImputer()),

        ('encoder', encoder)

    ])
from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline)

])
X_train = full_pipeline.fit_transform(train_data)

y_train = train_data['Survived']
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



rf_clf = RandomForestClassifier(random_state=11)

rf_clf.fit(X_train, y_train)

cross_val_score(rf_clf, X_train, y_train, cv=10).mean()
from sklearn.model_selection import GridSearchCV



param_grid = { 

    'n_estimators': [100, 200, 300, 400],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [2, 4, 6, 8],

    'criterion' :['gini', 'entropy']

}



grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5)

grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
from sklearn.metrics import accuracy_score



X_test = full_pipeline.fit_transform(test_data)

PassengerId = test_data['PassengerId']



y_pred = grid_search.predict(X_test)

submission = pd.DataFrame({'PassengerId' : PassengerId,

                          'Survived' : y_pred})

submission.head()