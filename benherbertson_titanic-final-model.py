import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

combine = pd.concat([train.drop('Survived', axis=1), test])



X_train = train.drop(['Survived'], axis=1)

y_train = train['Survived'].copy()
# Custom transformer for creating new feature `CabinMissing`

from sklearn.base import BaseEstimator, TransformerMixin



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_cabin_missing = True): # hyperparameter for tuning later

        self.add_cabin_missing = add_cabin_missing

    def fit(self, X, y=None):

        return self # nothing else to do

    def transform(self, X, y=None):

        if self.add_cabin_missing:

            X = X.to_numpy()

            cabin_array = X[:, 0]

            CabinMissing = pd.isnull(cabin_array)

            return np.c_[CabinMissing]

        else:

            return np.c_[X]
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')), # impute missing values with median

    ('minmax_scaler', MinMaxScaler()),             # scale features

])
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')), # impute missing values with mode

    ('cat_encoder', OneHotEncoder()),                     # convert text to numbers

])
from sklearn.compose import ColumnTransformer



num_attribs_all = combine.select_dtypes(['float64', 'int64']).columns

num_attribs = num_attribs_all.drop(['PassengerId'])



cat_attribs_all = combine.select_dtypes('object').columns

cat_attribs = cat_attribs_all.drop(['Ticket', 'Cabin', 'Name'])

cabin = ['Cabin']



full_pipeline = ColumnTransformer([

    # ('attribs_adder', CombinedAttributesAdder(add_cabin_missing=True), cabin), # create new feature `CabinMissing`

    ('num', num_pipeline, num_attribs),

    ('cat', cat_pipeline, cat_attribs),

    # ('drop', "drop", cabin),

])
X_train_prepared = full_pipeline.fit_transform(X_train)
from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier()

gb_clf.fit(X_train_prepared, y_train);
gb_clf.score(X_train_prepared, y_train)
from sklearn.model_selection import cross_val_score



accuracy_cv = cross_val_score(gb_clf, X_train_prepared, y_train, cv=5, scoring='accuracy')

np.mean(accuracy_cv)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30, 100, 150, 200], 'max_depth': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10]},

]



grid_search = GridSearchCV(gb_clf, param_grid, cv=5, scoring='accuracy', return_train_score=True)



grid_search.fit(X_train_prepared, y_train)
grid_search.best_params_
grid_search.best_estimator_
final_model = grid_search.best_estimator_



X_test = test.copy()

X_test_prepared = full_pipeline.transform(X_test)
# final_predictions = final_model.predict(X_test_prepared) # fine-tuned model

final_predictions = gb_clf.predict(X_test_prepared)        # model with default hyperparameter values
submit = test.copy()

submit['Survived'] = final_predictions

submit = submit[['PassengerId', 'Survived']]

submit.shape
submit.head()
submit.to_csv("../working/submit.csv", index=False)