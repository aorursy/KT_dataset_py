
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


df = pd.read_csv('../input/Train_ServicesOptedFor.csv')
df2 = pd.read_csv('../input/Train_Demographics.csv')
df3 = pd.read_csv('../input/Train (3).csv')

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
#Reformat Customer data by categories of service
customer_data = df.set_index(['CustomerID','TypeOfService']).unstack()
customer_data.columns = customer_data.columns.map('_'.join)

#Reindex for merge 
customer_data = customer_data.reset_index()

# Merge Dataframes together
merge_one = pd.merge(customer_data, df2, left_on='CustomerID', right_on='HouseholdID', how='outer').drop('HouseholdID', axis=1)
final = pd.merge(merge_one, df3, on='CustomerID', how='outer')

#Search for NAN values
final.HasDependents.isnull().values.any()

train_cats(final)
df, y, nas = proc_df(final, 'Churn')
m = RandomForestClassifier(n_estimators=50, min_samples_leaf = 3, max_features= 0.6, n_jobs=-1)
m.fit(df, y)
m.score(df,y)
def  split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 1000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(final, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestClassifier(n_estimators=50, min_samples_leaf = 3, max_features= 0.6, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
m  = RandomForestClassifier (n_estimators=50, min_samples_leaf = 3, max_features= 0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

