import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
data = pd.read_csv("/kaggle/input/housedata/output.csv", low_memory=False , parse_dates = ["date"])
display_all(data.T)# .T(transpoze)
display_all(data.price.describe())
print(sum(data["price"] == 0))
for i in range(data.columns.shape[0]):
    print(data.columns[i],"feature has {} null values".format(sum(data[data.columns[i]].isnull())))
data["price_na"] = data.price == 0 
sum(data["price_na"] == True)
data.loc[data.price_na == True].head()
data.loc[data.price == 0,["price"]] = data.price.median()
display_all(data.describe())
print(sum(data["price"] == 0))
data["price"]
data["price"] = np.log(data["price"])
data["price"]
data["date"].dt
data["date"].dt.year
import re
#regex library
def fixdatecolumn(data,column_name,drop = True, time = True):
    data_column = data[column_name]
    column_dtype = data_column.dtype
    
    targ_name = re.sub('[Dd]ate$', '', column_name)
    
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    for a in attr: 
        data[targ_name + a] = getattr(data_column.dt, a.lower())
        
    data[targ_name + 'Elapsed'] = data_column.astype(np.int64) // 10 ** 9
    
    if drop: 
        data.drop(column_name, axis=1, inplace=True)
fixdatecolumn(data,"date")

display_all(data.head())

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_bool_dtype
cat_list = []
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c) or is_bool_dtype(c):# Accualy for random forest, converting boolean dtype to category is unnecessary
            df[n] = c.astype("category").cat.as_ordered()
            cat_list.append(n)
print(cat_list)
def apply_cats(df, train):
    for n, c in df.items():
        if train[n].dtype == "category":
            df[n] = pd.Categorical(c, categories = train[n].cat.categories, ordered = True)
            
train_cats(data)
data["city"].cat.categories
data["city"].cat.codes
data_for_ohed = data.copy()
for n, c in data.items():
        
    if not is_numeric_dtype(c):
        data[n] = c.cat.codes+1
y = data.price
display_all(y.head())
x = data.drop(["price"], axis=1)
display_all(x.head())
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn import metrics
m = RandomForestRegressor(n_jobs=-1) # n_jobs =-1 allows us to use all units of processor
m.fit(x, y)
m.score(x,y)
def split_train_val(df,n): 
    return df[:n].copy(), df[n:].copy()
n_valid = int(np.floor(len(data)*0.2))
n_train = len(data)-n_valid
X_train, X_valid = split_train_val(x, n_train)
y_train, y_valid = split_train_val(y, n_train)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

import math
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())
def print_score(m):
    
    print(f"RMSE of train set {rmse(m.predict(X_train), y_train)}")
    print(f"RMSE of validation set {rmse(m.predict(X_valid), y_valid)}")
    print(f"R^2 of train set {m.score(X_train, y_train)}")
    print(f"R^2 of validation set {m.score(X_valid, y_valid)}")
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
best_result = 0
best_couple = [0,0]
for i in range(10,160,50):
    for j in range(1,15):
    
        m = RandomForestRegressor(n_estimators=i, min_samples_leaf=j, n_jobs=-1)
        m.fit(X_train, y_train)
    
        if m.score(X_valid, y_valid) > best_result :
            best_result = m.score(X_valid, y_valid)
            best_couple = [i,j]
    print("{}. epoch,choosing best".format(i/50+0.8))
m = RandomForestRegressor(n_estimators=best_couple[0], min_samples_leaf=best_couple[1], n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=best_couple[0], min_samples_leaf=best_couple[1], n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
import sklearn.ensemble 
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
set_rf_samples(1000)
m = RandomForestRegressor(n_estimators=best_couple[0], min_samples_leaf=best_couple[1], max_features=0.5, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
for i in cat_list:
    print(i,"unique values",x[i].nunique())
fi = pd.DataFrame({'columns':X_train.columns, 'importance':m.feature_importances_}
                       ).sort_values('importance', ascending=False)
fi.plot('columns', 'importance', 'barh', figsize=(12,7), legend=False)
keep_columns = fi[fi["importance"]<0.005]["columns"]; 
keep_columns
print(X_train.columns)
for i in X_train.columns :
    flag = 0
    for j in keep_columns:
        if i == j:
            flag = 1
    if flag == 1:    
        X_train.drop([i], axis=1, inplace=True)
        X_valid.drop([i], axis=1, inplace=True)
        
        
display_all(X_train)
m = RandomForestRegressor(n_estimators=best_couple[0], min_samples_leaf=best_couple[1], max_features=0.5, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
