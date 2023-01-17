# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%load_ext autoreload

%autoreload 2

%matplotlib inline
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

import math

import graphviz

import scipy

# import ggplot



# from pandas_summary import DataFrameSummary

from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn.tree import export_graphviz

from sklearn.ensemble import forest

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, StandardScaler

from scipy.cluster import hierarchy as hc

from pdpbox import pdp

from plotnine import *

# from concurrent.futures import ProcessPoolExecutor



from sklearn import metrics
# ??display # Uncomment for Documentation 
def add_datepart(df, fldname, drop = True):

    fld = df[fldname]

    if not np.issubdtype(fld.dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format = True)

    targ_pre = re.sub("[Dd]ate$", '', fldname)

    for n in ('Year', 'Month', 'Week', 'DayofWeek', 'DayofYear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):

        df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elasped'] = fld.astype(np.int64) // 10**9

    if drop: df.drop(fldname, axis = 1, inplace = True)
def apply_cats(df, trn):

    for n, c in df.items():

        if trn[n].dtype.name == "category":

            df[n] = pd.Categorical(c, categories = trn[n].cat.categories, ordered = True )
def train_cats(df):

    for n,c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict): 

            df[name + '_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None, prepoc_fn=None, max_n_cat=None,

           subset=None, mapper=None):

    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset:

        df = get_sample(df, subset)

    else:

        df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if prepoc_fn: prepoc_fn(df)

    if y_fld is None: y=None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds, axis=1, inplace=True)

    

    if na_dict is None: na_dict = {}

    else: na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for n, c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    if do_scale: mapper = scale_vars(df, mapper)

    for n, c in df.items(): numericalize(df, c, n, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    if do_scale: res = res + [mapper]

    return res
def numericalize(df, col, name, max_n_cat):

    if not is_numeric_dtype(col) and (max_n_cat is None or col.nunique()>max_n_cat):

        df[name] = col.cat.codes+1
def split_vals(a, n):

    return a[:n].copy(), a[n:].copy()
def get_sample(df, n):

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()
def set_rf_samples(n):

    forest._generate_sample_indices = (lambda rs, n_samples:

                                      forest.check_random_state(rs).randit(0, n_samples, n))
def reset_rf_samples():

    forest._generate_sample_indices = (lambda rs, n_samples:

                                      forest.check_random_state(rs).randit(0, n_samples, n_samples))
def scale_vars(df, mapper):

    warnings.filterwarnings("ignore", category = sklearn.exceptions.DataConversionWarning)

    if mapper is None:

        map_f = [([n], StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]

        mapper = DataFrameMapper(map_f).fit(df)

    df[mapper.transformed_names_] = mapper.transform(df)

    return mapper
def rmse(x, y):

    return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train),

          rmse(m.predict(X_valid), y_valid),

          m.score(X_train, y_train),

          m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'):

        res.append(m.oob_score_)

    print(res)
df_raw = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/Train.zip', low_memory = False, parse_dates = ["saledate"])
df_raw.SalePrice
def display_all(df):

    with pd.option_context("display.max_rows", 1000):

        with pd.option_context("display.max_columns", 1000):

            display(df)

            

display_all(df_raw.tail().transpose())
df_raw.SalePrice = np.log(df_raw.SalePrice)
df_raw.SalePrice
df_raw.saledate
fld = df_raw.saledate

fld.dt.year
add_datepart(df_raw, "saledate")

df_raw.saleYear.head()
df_raw.columns
df_raw.head()
train_cats(df_raw)
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(["High", "Medium", "Low"], ordered = True, inplace = True)
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
os.makedirs('tmp', exist_ok = True)

df_raw.to_feather('tmp/raw')
df_raw = pd.read_feather('tmp/raw')

df_trn, y_trn, nas = proc_df(df_raw, "SalePrice")
n_valid = 12000

n_trn = len(df_trn) - n_valid

X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)



raw_train, raw_valid = split_vals(df_raw, n_trn)
df_raw
set_rf_samples(50000)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)

%prun m.fit(X_train, y_train)

print_score(m)
%time preds = np.stack([t.predict(X_valid) for t in m.estimators_])

np.mean(preds[:, 0]), np.std(preds[:, 0])
x = raw_valid.copy()

x['pred_std'] = np.std(preds, axis=0)

x['pred'] = np.mean(preds, axis=0)

x.Enclosure.value_counts().plot.barh()
flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']

enc_sum = x[flds].groupby("Enclosure", as_index=False).mean()

enc_sum
enc_sum = enc_sum[~pd.isnull(enc_sum.SalePrice)]

enc_sum.plot("Enclosure", "SalePrice", "barh", xlim=(0, 11))
enc_sum.plot("Enclosure", "pred", "barh", xerr='pred_std', alpha=0.7, xlim=(0, 11))
raw_valid.ProductSize.value_counts().plot.barh()
flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']

summ = x[flds].groupby("ProductSize").mean()

summ
(summ.pred_std / summ.pred).sort_values(ascending=False)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df_trn)

fi[:10]
fi.plot('cols', 'imp', figsize=(10, 6), legend=False)
def plot_fi(fi):

    return fi.plot("cols", 'imp', 'barh', figsize=(12, 7), legend=False)
plot_fi(fi[:30])
to_keep = fi[fi.imp > 0.005].cols

len(to_keep)
df_keep = df_trn[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features = 0.5, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_keep)

plot_fi(fi)
df_trn2, y_trn, nas = proc_df(df_raw, "SalePrice", max_n_cat = 7)

X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_trn2)

plot_fi(fi[:25])
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1 - corr)

z = hc.linkage(corr_condensed, method="average")

fig = plt.figure(figsize=(16, 10))

dendrogram = hc.dendrogram(z, labels = list(df_keep.columns), orientation = "left", leaf_font_size=16)

plt.show()
def get_oob(df):

    m = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=5, max_features=0.5, oob_score=True)

    x, _ = split_vals(df, n_trn)

    m.fit(x, y_train)

    return m.oob_score_
get_oob(df_keep)
df = df_keep.copy()
for c in ('saleYear', 'saleElasped', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):

    print(c, get_oob(df.drop(c, axis=1)))
df1 = df_keep.copy()

to_drop = ['saleYear', 'fiModelDesc', 'Grouser_Tracks']

get_oob(df1.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)

X_train, X_valid = split_vals(df_keep, n_trn)
np.save('tmp/keep_cols.npy', np.array(df_keep.columns))
keep_cols = np.load('tmp/keep_cols.npy', allow_pickle=True)

df_keep = df_trn[keep_cols]
reset_rf_samples()
m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.5, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
set_rf_samples(50000)
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)

X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.6)

m.fit(X_train, y_train)

# print_score(m)
plot_fi(rf_feat_importance(m, df_trn2)[:10])
df_raw.plot("YearMade", "saleElasped", "scatter", alpha=0.01, figsize=(10,8))
x_all = get_sample(df_raw[df_raw.YearMade > 1930], 500)

# ggplot(x_all, aes('YearMade', 'SalePrice')) + stat_smooth(se=True, method="loess")
x = get_sample(X_train[X_train.YearMade > 1930], 500)
def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(m, x, feature=feat, model_features=x.columns)

    return pdp.pdp_plot(p, feat_name, plot_lines=True,

                       cluster = clusters is not None,

                       n_cluster_centers = clusters)
plot_pdp('YearMade')
plot_pdp('YearMade', clusters=5)
plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')
df_raw.YearMade[df_raw.YearMade < 1950] = 1950

df_keep["age"] = df_raw["age"] = df_raw.saleYear - df_raw.YearMade
X_train, X_valid = split_vals(df_keep, n_trn)

m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.6)

m.fit(X_train, y_train)

plot_fi(rf_feat_importance(m, df_keep))
!pip install treeinterpreter
from treeinterpreter import treeinterpreter as ti
df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)
row = X_valid.values[None, 0]

row
prediction, bias, contributions = ti.predict(m, row)
prediction[0], bias[0]
idxs = np.argsort(contributions[0])
[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]
contributions[0].sum()
df_ext = df_keep.copy()

df_ext['is_valid'] = 1

df_ext.is_valid[:n_trn] = 0

x, y, nas = proc_df(df_ext, 'is_valid')
m = RandomForestClassifier(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.6, oob_score=True)

m.fit(x, y)

m.oob_score_
fi = rf_feat_importance(m, x)

fi[:10]
feats = ['SalesID', 'saleElasped', 'MachineID']
(X_train[feats]/1000).describe()
(X_valid[feats]/1000).describe()

x.drop(feats, axis=1, inplace=True)
m = RandomForestClassifier(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.6, oob_score=True)

m.fit(x, y)

m.oob_score_
fi = rf_feat_importance(m, x)

fi[:10]
set_rf_samples(50000)
feats = ['SalesID', 'saleElasped', 'MachineID', 'age', 'YearMade', 'saleDayofYear']
X_train, X_valid = split_vals(df_keep, n_trn)

m = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=3, max_features=0.6, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
df = df_keep.copy()
for f in feats:

    df_subs = df.drop(f, axis=1)

    X_train, X_valid = split_vals(df_subs, n_trn)

    m = RandomForestRegressor(n_estimators=40, n_jobs=-1, min_samples_leaf=3, max_features=0.6, oob_score = True)

    m.fit(X_train, y_train)

    print(f)

    print_score(m)
df = df_keep.copy()
df_subs = df.drop(['SalesID', 'MachineID', 'saleDayofYear'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)

m = RandomForestRegressor(n_estimators=80, n_jobs=-1, min_samples_leaf=3, max_features=0.6, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
plot_fi(rf_feat_importance(m, X_train))
m = RandomForestRegressor(n_estimators=160, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)