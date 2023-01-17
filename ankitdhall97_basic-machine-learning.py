import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

from xgboost import XGBClassifier

import os
#print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.shape
train_data.head()
train_data.info()
train_data.describe()
train_data.isnull().sum()
del train_data['Cabin']
# Inspired from stackoverflow.com/questions/25239958

from sklearn.base import BaseEstimator, TransformerMixin

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



num_attributes = ['Age','SibSp','Parch','Fare']

cat_attributes = ['Pclass','Sex','Embarked']



num_pipeline = Pipeline([('imputer',SimpleImputer(strategy='median')),

                        ('std_scalar',StandardScaler())])

cat_pipeline = Pipeline([('imputer',MostFrequentImputer()),

                         ('label_enc',OrdinalEncoder()),

                         ('onehot',OneHotEncoder())])

full_pipeline = ColumnTransformer([('num',num_pipeline,num_attributes),

                                  ('cat',cat_pipeline,cat_attributes)])
X_train = full_pipeline.fit_transform(train_data)

X_train
y_train = train_data['Survived']
from sklearn.svm import SVC



svm_clf = SVC(gamma="auto", probability=True)

svm_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score



svm_scores = cross_val_score(svm_clf,X_train,y_train,cv=10)
svm_scores.mean()
X_test = full_pipeline.transform(test_data)

y_pred = svm_clf.predict(X_test)
my_submission_svc = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

my_submission_svc.to_csv('submission_svc.csv', index=False)
kfold = StratifiedKFold(n_splits=10)
svc = SVC(probability=True)

param_grid_svc = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}

grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=10, refit=True, verbose=1, n_jobs= 4)

grid_search_svc.fit(X_train, y_train)
grid_search_svc.best_params_
grid_search_svc.best_estimator_

grid_search_svc.best_score_
svc_model = grid_search_svc.best_estimator_
etc = ExtraTreesClassifier()
param_grid_etc = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}

grid_search_etc = GridSearchCV(etc, param_grid_etc, cv=kfold, n_jobs=4)

grid_search_etc.fit(X_train, y_train)
print(grid_search_etc.best_params_)

print(grid_search_etc.best_estimator_)

print(grid_search_etc.best_score_)

etc_model = grid_search_etc.best_estimator_
rfc = RandomForestClassifier()
param_grid_rfc = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}

grid_search_rfc = GridSearchCV(rfc, param_grid_rfc, cv=kfold, n_jobs=4)

grid_search_rfc.fit(X_train, y_train)
print(grid_search_rfc.best_params_)

print(grid_search_rfc.best_estimator_)

print(grid_search_rfc.best_score_)

rfc_model = grid_search_rfc.best_estimator_
voting = VotingClassifier(estimators=[('rfc', rfc_model), ('extc', etc_model),

('svc', svc_model)], voting='soft', n_jobs=4)



voting = voting.fit(X_train, y_train)
y_preds_voting = voting.predict(X_test)
my_submission_ensemble = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_preds_voting})

my_submission_ensemble.to_csv('my_submission_ensemble.csv', index=False)
xgb = XGBClassifier(max_depth = 5, eta = 0.1, gamma = 0.1, colsample_bytree = 1, min_child_weight = 1,

    n_estimators = 500)
xgb.fit(X_train, y_train)
y_preds_xgb = xgb.predict(X_test)
my_submission_xgb = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_preds_xgb})

my_submission_xgb.to_csv('my_submission_xgb.csv', index=False)