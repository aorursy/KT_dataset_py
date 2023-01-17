%load_ext autoreload
%autoreload 2

%matplotlib inline
import pandas as pd
import numpy as np

import re
import os
import math

from sklearn.ensemble import RandomForestRegressor

from pandas.api.types import is_string_dtype, is_numeric_dtype
PATH = '../input/'
!ls {PATH}
train_df = pd.read_csv(f'{PATH}train/Train.csv', low_memory=False, parse_dates=['saledate'])
valid_df = pd.read_csv(f'{PATH}valid/Valid.csv', low_memory=False, parse_dates=['saledate'])
def display_all(df):
    with pd.option_context('display.max_rows',1000, 'display.max_columns',100):
        display(df)
display_all(train_df.tail().T)
display_all(train_df.describe(include='all').T)
train_df.SalePrice = np.log(train_df.SalePrice)
train_df.SalePrice[:5]
def add_datepart(df, fldname, drop=True):
    fld = df[fldname]
    
    # if fld is not of type datetime convert it to datetime
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        
    targ_pre = re.sub('[Dd]ate$','',fldname)
    
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 
             'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt, n.lower())
        
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop:
        df.drop(fldname, axis=1, inplace=True)
add_datepart(train_df, 'saledate')
add_datepart(valid_df, 'saledate')
train_df.saleYear.head()
def train_cats(df):
    for n, c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
            
def apply_cats(df, train):
    for n, c in df.items():
        if (n in train.columns) and (train[n].dtype.name == 'category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(train[n].cat.categories, ordered=True, inplace=True)
train_cats(train_df)
apply_cats(valid_df, train_df)
train_df.UsageBand.cat.categories
train_df.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
valid_df.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
display_all(train_df.isnull().sum())
print('--------------------------------------')
display_all(valid_df.isnull().sum())
os.makedirs('tmp', exist_ok=True)
train_df.to_feather('tmp/bulldozers-train-raw')
valid_df.to_feather('tmp/bulldozers-valid-raw')
train_df = pd.read_feather('tmp/bulldozers-train-raw')
valid_df = pd.read_feather('tmp/bulldozers-valid-raw')
def fix_missing(df, col, name):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum():
            #df[name+'_na'] = pd.isnull(col)
            filler = col.median()
            df[name] = col.fillna(filler)
            #na_dict[name] = filler
def numericalize(df, col, name):
    if not is_numeric_dtype(col):
        df[name] = col.cat.codes + 1
def proc_df(df, y_fld=None):
    
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]) : df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        df.drop(y_fld, axis=1, inplace=True)
    
    for n,c in df.items(): fix_missing(df, c, n)
    for n,c in df.items(): numericalize(df, c, n)
        
    df = pd.get_dummies(df, dummy_na=True)
    
    return [df,y]
train, y = proc_df(train_df, 'SalePrice')
test, temp = proc_df(valid_df)
def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 12000
n_train = len(train) - n_valid

X_train, X_valid = split_vals(train, n_train)
y_train, y_valid = split_vals(y, n_train)
X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print(res)
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
y_pred = m.predict(test)
Submission = pd.DataFrame({'SalesID':valid_df.SalesID, 'SalePrice':y_pred})
Submission.to_csv('Submission.csv', index=False)