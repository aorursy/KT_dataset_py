# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.info()
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
train_cats(train)
train.BldgType.cat.categories
train.isnull().sum().sort_index()/len(train)
#The below functions are taken from fast ai library, code for which can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes+1
        
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res
def len_cat(df):
    for n,c in df.items():
        if not is_numeric_dtype(df[n]):
            print(f'Colum Name: {n}, Length of Category: {len(df[n].cat.categories)}')
len_cat(train)
df, y, nas = proc_df(train, 'SalePrice', skip_flds=["Id", "Alley", "Fence", "MiscFeature", "PoolQC"]
                    , max_n_cat = 9)
y = np.log(y)
def split_vals(a,n): return a[:n], a[n:]
df.shape
n_valid = int(1460 * 0.1)
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
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, max_features=0.5, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df)
fi[:10]
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30])
sns.boxplot(x='OverallQual', y='SalePrice', data = train)
plt.scatter(train["GrLivArea"], train["SalePrice"])
plt.show()
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=train)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
def apply_cats(df, trn):
     for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)
test.shape
apply_cats(test, train)
test.BldgType.cat.categories
df_test, _ , nas = proc_df(test, None, skip_flds=["Id", "Alley", "Fence", "MiscFeature", "PoolQC"]
                    , max_n_cat = 9, na_dict=nas)
df_test.head()
columns = set(df.columns).intersection(set(df_test.columns))
drop_columns = []
for n, c in df_test.items():
    if n not in columns:
        drop_columns.append(n)
df_test.drop(drop_columns, axis = 1, inplace=True)
df_test.shape
y_test = np.exp(m.predict(df_test))
#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_test})
#my_submission.to_csv('submission.csv', index=False)
xg_reg = xgb.XGBRegressor(learning_rate = 0.11, n_estimators = 300)
xg_reg.fit(X_train, y_train)
print_score(xg_reg)
y_test = np.exp(m.predict(df_test))
#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_test})
#my_submission.to_csv('submission_xgboost.csv', index=False)
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(xg_reg, max_num_features=30, ax =ax)
