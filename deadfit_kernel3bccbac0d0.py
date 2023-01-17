import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from numba import jit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit

plt.style.use('ggplot')
x = pd.read_csv('../input/dataset_train.csv', index_col = 0, parse_dates=['date'])
y = x.sales.to_frame()
x_true = x.drop("sales", axis = 1)
x_true = x_true.reset_index()
y = y.reset_index()
tscv = TimeSeriesSplit(n_splits = 60)

for train_index, test_index in tscv.split(x_true):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = x_true.loc[train_index], x_true.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

x_true = x_true.reset_index()
y = y.reset_index()
from catboost import CatBoostRegressor, CatBoostClassifier, FeaturesData, Pool, MetricVisualizer
date_train = X_train["date"]
date_test = X_test["date"]
X_train = X_train.drop("date", axis = 1)
X_test = X_test.drop("date", axis = 1)
y_train = y_train.drop("date", axis = 1)
y_test = y_test.drop("date", axis = 1)
X_train['year'] = date_train.dt.year
X_train['month'] = date_train.dt.month
X_train['weekofyear'] = date_train.dt.weekofyear
X_train['dayofyear'] = date_train.dt.dayofyear

X_test['month'] = date_test.dt.month
X_test['weekofyear'] = date_test.dt.weekofyear
X_test['year'] = date_test.dt.year
X_test['dayofyear'] = date_test.dt.dayofyear
clf2 = CatBoostRegressor(iterations = 200, l2_leaf_reg = 8, learning_rate = 0.15, depth = 10, random_seed = 288) 
clf2.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, plot = True)
test = pd.read_csv('../input/dataset_valid.csv', parse_dates=['date'])

test['month'] = test.date.dt.month
test['weekofyear'] = test.date.dt.weekofyear
test['year'] = test.date.dt.year
test['dayofyear'] = test.date.dt.dayofyear
test['weekday'] = test.date.dt.weekday

test.head()
test = test.reset_index()
y_submission = pd.read_csv('../input/sample_submission.csv', index_col=0)
test = test.drop("date", axis = 1)
y_submission = clf2.predict(test)
import time
import os

current_timestamp = int(time.time())
submission_path = 'submissions/{}.csv'.format(current_timestamp)

if not os.path.exists('submissions'):
    os.makedirs('submissions')
test = test.reset_index()
print(submission_path)
sub = pd.DataFrame({"id": test.index, "sales": y_submission})
sub.to_csv(submission_path, index = False)
