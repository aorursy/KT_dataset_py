from fastai.imports import *

#from fastai.structured import *

from pandas.api.types import is_string_dtype, is_numeric_dtype



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics



from pdpbox import pdp

from plotnine import *
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_raw = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_raw.SalePrice = np.log(df_raw.SalePrice)
# def add_datepart(df, fldname, drop=True):

#    fld = df[fldname]

#     if not np.issubdtype(fld.dtype, np.datetime64):

#         df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

#     targ_pre = re.sub('[Dd]ate$', '', fldname)

#     for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

#             'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):

#         df[targ_pre+n] = getattr(fld.dt,n.lower())

#     df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

#     if drop: df.drop(fldname, axis=1, inplace=True)
# add_datepart(df_raw, 'saledate')
def train_cats(df):

    for n,c in df.items():

            if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
def apply_cats(df, trn):

    for n,c in df.items():

        if (n in trn.columns) and (trn[n].dtype.name=='category'):

            df[n] = c.astype('category').cat.as_ordered()

            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)
train_cats(df_raw) #make categories out of strings

apply_cats(df=df_test, trn=df_raw)
def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict
def get_sample(df,n):

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()
def scale_vars(df, mapper):

    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)

    if mapper is None:

        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]

        mapper = DataFrameMapper(map_f).fit(df)

    df[mapper.transformed_names_] = mapper.transform(df)

    return mapper
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
def numericalize(df, col, name, max_n_cat):

   if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique()>max_n_cat):

        df[name] = col.cat.codes+1
df, y, nas = proc_df(df_raw, 'SalePrice')

X_test, _, nas = proc_df(df_test, na_dict=nas)
df.T.iloc[:, :10]
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 500  

n_trn = len(df)-n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

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
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df); fi[:10]
m = RandomForestRegressor(n_jobs=-1, n_estimators=50, max_features=0.5, min_samples_leaf=2)

%time m.fit(X_train, y_train)

print_score(m)
#to_keep = fi[fi.imp>0.005].cols; len(to_keep)
#df_keep = df[to_keep].copy()

#X_train, X_valid = split_vals(df_keep, n_trn)
# m = RandomForestRegressor(n_jobs=-1, n_estimators=50, max_features=0.5, min_samples_leaf=2)

# %time m.fit(X_train, y_train)

# print_score(m)
predictions = np.exp(m.predict(X_test.values))
sub = pd.DataFrame()

sub['Id'] = df_test.Id

sub['SalePrice'] = predictions

sub.to_csv('submission4.csv',index=False)