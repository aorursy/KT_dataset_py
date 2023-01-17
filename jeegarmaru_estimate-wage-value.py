# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



fifa = pd.read_csv('../input/data.csv', index_col='ID')

fifa.head()



# Any results you write to the current directory are saved as output.
fifa = fifa.drop('Unnamed: 0,Name,Photo,Nationality,Flag,Club,Club Logo,Jersey Number,Joined,Real Face,Loaned From,Release Clause'.split(','), axis=1)

fifa.head()
def parse_money(s):

    if s.startswith('€'):

        s = s[1:]

    multiplier = None

    if s.endswith('M'):

        s = s[:-1]

        multiplier = 1e6

    elif s.endswith('B'):

        s = s[:-1]

        multiplier = 1e9

    elif s.endswith('K'):

        s = s[:-1]

        multiplier = 1e3

    f = float(s)

    if multiplier:

        f = f * multiplier

    return f
fifa['Value'] = fifa['Value'].apply(parse_money)

fifa['Wage'] = fifa['Wage'].apply(parse_money)

fifa[['Value', 'Wage']].head()
fifa[['Work Rate1', 'Work Rate2']] = fifa['Work Rate'].str.split('/', expand=True)

fifa = fifa.drop('Work Rate', axis=1)

fifa[['Work Rate1', 'Work Rate2']].head()
def parse_date(s):

    if isinstance(s, str) and ',' in s:

        return float(s.split()[2])

    else:

        return float(s)

fifa['Contract Valid Until'] = fifa['Contract Valid Until'].apply(parse_date)

fifa['Contract Valid Until'].head()
def parse_height(s):

    if isinstance(s, float):

        return s

    f, i = s.split("'")

    return int(f)*12 + int(i)

fifa['Height'] = fifa['Height'].apply(parse_height)

fifa['Height'].head()
def parse_weight(s):

    if isinstance(s, str) and s.endswith('lbs'):

        return float(s[:-3])

    return float(s)

fifa['Weight'] = fifa['Weight'].apply(parse_weight)

fifa['Weight'].head()
rating_cols = "LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB".split(', ')

for col in rating_cols:

    fifa[[f"{col}1", f"{col}2"]] = fifa[col].str.split('+', expand=True)

    fifa[f"{col}1"] = fifa[f"{col}1"].astype('float')

    fifa[f"{col}2"] = fifa[f"{col}2"].astype('float')

    fifa = fifa.drop(col, axis=1)

fifa.head()
def null_counts(df):

    for col in df.columns:

        print(f"{col} : {df[col].isnull().sum()}")

null_counts(fifa)
fifa = fifa[~fifa['ShortPassing'].isnull()]

null_counts(fifa)
fifa.loc[fifa['Position'].isnull(), 'Position'] = 'Unknown'

null_counts(fifa)
cat_columns = fifa.select_dtypes('object', 'category').columns

null_counts(fifa[cat_columns])
fifa['Body Type'].value_counts()
fifa.loc[~fifa['Body Type'].isin(['Normal', 'Stocky', 'Lean']), 'Body Type'] = fifa['Body Type'].value_counts().index[0]

fifa['Body Type'].value_counts()
categoricals = fifa.select_dtypes('object')

for col in categoricals.columns:

    print(col)

    print("----------------------------")

    print(fifa[col].value_counts())
from sklearn.model_selection import train_test_split

train, test = train_test_split(fifa, test_size=0.2, random_state=42)
labels = ['Value', 'Wage']

X_train, X_test = (X.drop(labels=labels, axis=1, inplace=False) for X in (train, test))

y_train, y_test = (X[labels].copy() for X in (train, test))

y_test.columns
cat_columns = X_train.select_dtypes('object', 'category').columns

num_columns = X_train.select_dtypes(exclude=['object', 'category']).columns
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):

        self.feature_names = feature_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.feature_names].values

    def get_feature_names(self):

        return self.feature_names
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

cat_selector = DataFrameSelector(cat_columns)

ohe = OneHotEncoder(sparse=False)

cat_pipeline = Pipeline([

    ('selector', cat_selector),

    ('ohe', ohe)

])
from sklearn.preprocessing import Imputer, StandardScaler

num_selector = DataFrameSelector(num_columns)

num_pipeline = Pipeline([

    ('selector', num_selector),

    ('imputer', Imputer(strategy='median')),

    ('scaler', StandardScaler())

])
from sklearn.pipeline import FeatureUnion

pipeline = FeatureUnion([

    ('cat_pipeline', cat_pipeline),

    ('num_pipeline', num_pipeline)

])
X_train_arr = pipeline.fit_transform(X_train)

X_train_arr
feature_names = list(ohe.get_feature_names()) + list(num_selector.get_feature_names())
X_test_arr = pipeline.transform(X_test)

X_test_arr
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_arr, y_train)
def print_metrics(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("Mean Squared error : %s" % mean_squared_error(y_test, y_pred))

    print("Mean Absolute error : %s" % mean_absolute_error(y_test, y_pred))

    print("R2 Score: %s" % r2_score(y_test, y_pred))
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print_metrics(lr, X_test_arr, y_test)
from sklearn.linear_model import SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import LinearSVR

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

models = [SGDRegressor(), DecisionTreeRegressor(), LinearSVR(), AdaBoostRegressor(), 

          GradientBoostingRegressor(), RandomForestRegressor(), MLPRegressor()]



label = 'Wage'

for model in models:

    print(f"Running model {type(model)}")

    model.fit(X_train_arr, y_train[label])

    print_metrics(model, X_test_arr, y_test[label])

    print("------------------------")
def sort_rsearch_results(rsearch):

    cvres = rsearch.cv_results_

    rsearch_results = sorted([(np.sqrt(-x[0]), x[1]) for x in zip(cvres['mean_test_score'], cvres['params'])])

    return rsearch_results



def feature_importance(model):

    return sorted(zip(feature_names, model.feature_importances_), key=lambda x : x[1], reverse=True)
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV

rfr = RandomForestRegressor()

rfr_params = {'n_estimators' : [50, 100, 200], 'bootstrap' : [True, False], 

              'min_samples_split' : [0.1, 0.01, 0.001], 'max_depth' : [3, 5, 8],

              'max_features' : ['sqrt', 0.1, 0.01], 'warm_start' : [True, False]

          }

rsearch = RandomizedSearchCV(rfr, rfr_params, cv=8, n_iter=10, n_jobs=5, scoring='neg_mean_squared_error', verbose=True)

rsearch.fit(X_train_arr, y_train[label])
sort_rsearch_results(rsearch)
rfr_final = rsearch.best_estimator_

print_metrics(rfr_final, X_test_arr, y_test[label])
feature_importance(rfr_final)
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV

gbm = GradientBoostingRegressor(n_iter_no_change=10)

params = {'learning_rate' : [0.1, 0.01, 0.001], 'n_estimators' : [50, 100, 200],

           'subsample' : [0.8, 0.9, 1.0], 'min_samples_split' : [0.01, 0.001],

           'max_depth' : [3, 5, 8], 'max_features' : ['auto', 'sqrt', 'log2'],

          }

rsearch = RandomizedSearchCV(gbm, params, cv=8, n_iter=10, n_jobs=10, scoring='neg_mean_squared_error', verbose=True)

rsearch.fit(X_train_arr, y_train[label])
sort_rsearch_results(rsearch)
gbm_final = rsearch.best_estimator_

print_metrics(gbm_final, X_test_arr, y_test[label])
feature_importance(gbm_final)