import pandas as pd
import numpy as np
import math
import re

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
%matplotlib inline
path = '../input/train/'
!ls {path}
df_org = pd.read_csv(f'{path}Train.csv', low_memory=False, parse_dates=['saledate']); df_org.head().T
df_org.describe(include='all').T
df_org.SalePrice = np.log(df_org.SalePrice) #taking log, now over metric is simple RMSE instead of RMSLE.
m = RandomForestRegressor()
m.fit(df_org.drop('SalePrice', axis=1), df_org.SalePrice)
df_raw = df_org.copy() # keeping a copy of our original dataset aside
categorical = []
for col in df_raw.columns[:]:
    if df_raw[col].dtype == 'object' : categorical.append(col)  # pandas treat "str" as "object"
categorical # list of all the variables which are strings
for col in categorical: df_raw[col] = df_raw[col].astype("category").cat.as_ordered()
# you can select a column and have alook at the categories
df_raw.UsageBand.cat.categories
# We can specify the order to use for categorical variables if we wish:
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes
m = RandomForestRegressor()
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `fldname` of `df`."
    fld = df[fldname]
    fld_dtype = fld.dtype
    
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtypeType):
        fld_dtype = np.datetime64
        
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
         
    prefix = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: 
        df[prefix + n] = getattr(fld.dt, n.lower())
    df[prefix + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        
    if drop: df.drop(fldname, axis=1, inplace=True)
add_datepart(df_raw, 'saledate'); df_raw.head().T
df_raw.isnull().mean().sort_index()
def fix_missing(df, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing."""
    for name,col in df.items():
        if is_numeric_dtype(col):
            if pd.isnull(col).sum():
                df[name+'_na'] = pd.isnull(col)
                filler = na_dict[name] if name in na_dict else col.median()
                df[name] = col.fillna(filler)
                na_dict[name] = filler
    return na_dict
na_dict = fix_missing(df_raw, {})
def numericalize(df, max_cat):
    """ Changes the column col from a categorical type to it's integer codes."""
    for name, col in df.items():
        if hasattr(col, 'cat') and (max_cat is None or len(col.cat.categories)>max_cat):
            df[name] = col.cat.codes+1
            
def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement."""
#     idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[-n:].copy()
def process_df(df_raw,y_fld=None, subset=None, na_dict={}, max_cat=None,):
    if subset: df = get_sample(df_raw,subset)
    else: df = df_raw.copy()
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
    df.drop(y_fld, axis=1, inplace=True)
    
    # Missing continuous values
    na_dict = fix_missing(df, na_dict)
    
    # Normalizing continuous variables
    means, stds = {}, {}
    for name,col in df.items():
        if is_numeric_dtype(col) and col.dtype not in ['bool', 'object']:
            means[name], stds[name] = col.mean(), col.std()
            df[name] = (col-means[name])/stds[name] 
    
    # categorical variables
    categorical = []
    for col in df.columns:
        if df[col].dtype == 'object' : categorical.append(col)  # pandas treat "str" as "object"
    for col in categorical: 
        df[col] = df[col].astype("category").cat.as_ordered()
        
    # converting categorical variables to integer codes.
    numericalize(df, max_cat) # features with cardinality more than "max_cat".
    
    df = pd.get_dummies(df, dummy_na=True) # one-hot encoding for features with cardinality lower than "max_cat".
    
    return df, y#, na_dict, means, stds
add_datepart(df_org, 'saledate')
df, y = process_df(df_org,'SalePrice')
df.head().T # final dataset, you can save it if you want!
m = RandomForestRegressor(n_jobs=-1)
m.fit(df,y)
m.score(df, y)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
df, y = process_df(df_org, 'SalePrice', subset=30000)

X_train, X_valid = split_vals(df, 20000)
y_train, y_valid = split_vals(y, 20000) 
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
df, y = process_df(df_org,'SalePrice')

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds.shape
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
## Also a BASELINE MODEL to which all the other models will be compared.
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)
dectree_max_depth(m.estimators_[0].tree_)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
dectree_max_depth(m.estimators_[0].tree_)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
