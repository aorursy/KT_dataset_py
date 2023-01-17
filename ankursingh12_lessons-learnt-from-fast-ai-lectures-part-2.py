import pandas as pd
import numpy as np
import re
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
%matplotlib inline
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
df_org = pd.read_csv('../input/train/Train.csv', low_memory=False); df_org.head()
df_org.SalePrice = np.log(df_org.SalePrice)
df = df_org.copy()
add_datepart(df, 'saledate')
df, y = process_df(df, 'SalePrice'); df.head().T
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
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
def rf_feature_importance(m,df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
fi = rf_feature_importance(m, X_train); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.005].cols; len(to_keep) # taking only the important features
df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m) # removing less important features not just saved the computation but also gave better scores.
fi = rf_feature_importance(m, df_keep)
plot_fi(fi);
from scipy.cluster import hierarchy as hc
import scipy
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_
get_oob(df_keep)
for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))
to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)
keep_cols = df_keep.columns
df_keep = df[keep_cols]
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y = process_df(df_ext, 'is_valid')
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_
fi = rf_feature_importance(m, x); fi[:10]
feats=['SalesID', 'saleElapsed', 'MachineID']
(X_train[feats]/1000).describe()
(X_valid[feats]/1000).describe()
x.drop(feats, axis=1, inplace=True)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_
fi = rf_feature_importance(m, x); fi[:10]
feats=['SalesID', 'saleElapsed', 'MachineID', 'state', 'saleDay', 'saleDayofyear']
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)
df_subs = df_keep.drop(['SalesID', 'MachineID'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
plot_fi(rf_feature_importance(m, X_train));
# np.save('tmp/subs_cols.npy', np.array(df_subs.columns)) # save the final columsn list

from treeinterpreter import treeinterpreter as ti
df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)
row = X_valid.values[None,0]; row
prediction, bias, contributions = ti.predict(m, row)
prediction[0], bias[0]
idxs = np.argsort(contributions[0])
[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]
contributions[0].sum()
m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)