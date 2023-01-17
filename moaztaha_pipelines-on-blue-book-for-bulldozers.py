# Python >= 3.5 is required 

import sys

assert sys.version_info >= (3,5)



# Scikit-Learn >= 0.20 is required 

import sklearn

assert sklearn.__version__ >= "0.20"



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Ignore useless warnings 

import warnings

warnings.filterwarnings(action='ignore', message='^internal gelsd')
!ls ../input/blue-book-for-bulldozer/
# get the data

import pandas as pd

train_set = pd.read_csv('../input/blue-book-for-bulldozer/Train/Train.csv', delimiter=',')

valid_set = pd.read_csv('../input/blue-book-for-bulldozer/Valid/Valid.csv', delimiter=',')
train_set.head()
# nan values percentages per columns

nan_features_pct = {}

for feature, nans in zip(train_set.columns, train_set.isna().sum()):

    pct = nans / train_set.shape[0]

    if pct > 0:

        nan_features_pct[feature] = pct

nan_features_pct
train_set.dtypes
object_features = [feature for feature, dtype in zip(train_set.columns, train_set.dtypes) if dtype=='object']

object_features
from sklearn.base import BaseEstimator, TransformerMixin



class ObjectAndCategoricalAttributes(BaseEstimator, TransformerMixin):

    def __init__(self, add_objects=False, add_nans=False, date=True):

        self.add_objects = add_objects

        self.add_nans = add_nans

        self.date = date

        self.object_features = []

        self.nan_features = {}

        

    def fit(self, X, y=None):

        if not self.add_objects:

            self.object_features = [feature for feature, dtype in zip(X.columns, X.dtypes) if dtype=='object' and feature != 'saledate']

        

        if not self.add_nans:

            for feature, nans in zip(X.columns, X.isna().sum()):

                self.nan_features[feature] = nans/X[feature].shape[0]



        return self

    

    def transform(self, X):

        new_data = X.copy()

        if not self.add_objects:

            new_data.drop(self.object_features, axis=1, inplace=True)

        

        if not self.add_nans:

            for feature, nans in self.nan_features.items():

                if nans > 0:

                    try:

                        new_data.drop(feature, axis=1, inplace=True)

                    except KeyError:

                        continue

        

        return new_data
# Testing Transformer

cat_ob_transformer = ObjectAndCategoricalAttributes()

cat_ob_transformer.fit(train_set)

train_temp = cat_ob_transformer.transform(train_set)

train_temp.head()
class DatesHandler(BaseEstimator, TransformerMixin):

    def __init__(self, year=True, month=False, day=False):

        self.year = year,

        self.month = month

        self.day = day

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        X_new = X.copy()

        if self.year:

            X_new['year'] = [date.year for date in pd.to_datetime(X_new.saledate)]

        if self.month:

            X_new['month'] = [date.month for date in pd.to_datetime(X_new.saledate)]

        if self.day:

            X_new['day'] = [date.day for date in pd.to_datetime(X_new.saledate)]

            

        return X_new.drop('saledate', axis=1)
datesHandler = DatesHandler(1, 1, 1)

train_temp = datesHandler.transform(train_temp)

train_temp.head()
from sklearn.pipeline import Pipeline



base_pipeline = Pipeline([

    ('Objects_Nans', ObjectAndCategoricalAttributes()),

    ('DatesHandler', DatesHandler())

])
# loss function

from sklearn.metrics import mean_squared_log_error, make_scorer

import numpy as np

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_log_error(y, y_pred))
# Transform the data

base_pipeline.fit(train_set)

X = base_pipeline.transform(train_set)



y = X.SalePrice

X.drop('SalePrice', axis=1, inplace=True)
X.head()
y.head()
# No nans and spread the date

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



reg_rfg = RandomForestRegressor(random_state=42)

param_grid = {'bootstrap': [False, True], 'n_estimators': [3, 10], 'max_features': ['auto', 'sqrt']}



base_model = GridSearchCV(reg_rfg, param_grid, make_scorer(rmsle), cv=5, n_jobs=-1, verbose=1)



base_model.fit(X, y)
base_model.best_score_
# Expand the dates

base_pipeline = Pipeline([

    ('Objects_Nans', ObjectAndCategoricalAttributes()),

    ('DatesHandler', DatesHandler(1, 1, 1))

])
# Transform the data

base_pipeline.fit(train_set)

X = base_pipeline.transform(train_set)



y = X.SalePrice

X.drop('SalePrice', axis=1, inplace=True)
X.head()
base_model = GridSearchCV(reg_rfg, param_grid, make_scorer(rmsle), cv=5, n_jobs=-1, verbose=1)



base_model.fit(X, y)
base_model.best_score_
train_set['datasource'].hist()
train_set['ModelID'].hist()
train_set['MachineID'].hist()
train_set['YearMade'].hist()
from pandas.plotting import scatter_matrix



attributes = ['SalePrice', 'saledate']



scatter_matrix(train_set[attributes], figsize=(12, 8))

plt.plot()
train_set.plot(kind='scatter', x='SalePrice', y='YearMade', alpha=0.1)

plt.axis([100000, 150000, 1960, 2015])

plt.plot()
X.plot(kind='scatter', x='month', y='year', alpha=0.01)

#plt.axis([1960, 2015, 1960, 2015])

plt.plot()
X.plot(kind='scatter', x='day', y='year', alpha=0.01)

#plt.axis([1960, 2015, 1960, 2015])

plt.plot()
# Expand the dates -> only using the years

base_pipeline = Pipeline([

    ('Objects_Nans', ObjectAndCategoricalAttributes()),

    ('DatesHandler', DatesHandler())

])



base_pipeline.fit(train_set)

X = base_pipeline.transform(train_set)

y = X.SalePrice

X.drop('SalePrice', axis=1, inplace=True)
base_model = GridSearchCV(reg_rfg, param_grid, make_scorer(rmsle), cv=5, n_jobs=-1, verbose=1)



base_model.fit(X, y)

base_model.best_score_
test_df = pd.read_csv('../input/blue-book-for-bulldozer/Test.csv')

test_df.head()
X_test = base_pipeline.transform(test_df)

X_test.head()
test_preds = base_model.predict(X_test)

test_preds[:10]
submission = pd.DataFrame({'SalesID': test_df.SalesID, 'SalePrice': test_preds})

submission.head()
submission.to_csv('submission.csv', index=False)