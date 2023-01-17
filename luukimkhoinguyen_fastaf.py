!pip install treeinterpreter
!pip install scikit-misc
!pip install fastai==0.7.0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_summary import DataFrameSummary

from fastai.imports import *

from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

print('ass')



# Any results you write to the current directory are saved as output.
print(os.listdir("/kaggle/input/bluebookforbulldozers"))

PATH = "/kaggle/input/bluebookforbulldozers/"
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory = False, parse_dates = ['saledate'])
def display_all(df):

    with pd.option_context("display.max_rows", 1000):

        with pd.option_context("display.max_columns", 1000):

            display(df)
df_raw.head()
df_raw['SalePrice'].dtypes
df_raw.SalePrice = np.log(df_raw.SalePrice)

df_raw.SalePrice

fld = df_raw.saledate
add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()
train_cats(df_raw)
df_raw.UsageBand.cat.codes
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered = True, inplace = True)
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
os.makedirs('tmp', exist_ok = True)

df_raw.to_feather('tmp/raw')
df_raw = pd.read_feather('tmp/raw')
df, y, z = proc_df(df_raw, 'SalePrice')
df.head()
m = RandomForestRegressor(n_jobs = -1)

m.fit(df, y)

m.score(df, y)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 12000  # same as Kaggle's test set size

n_trn = len(df)-n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):

    res = [rmse(m.predict(X_train), y_train),

           rmse(m.predict(X_valid), y_valid),

           m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
df_trn, y_trn, z_trn = proc_df(df_raw, 'SalePrice', subset = 30000)

X_train, _ = split_vals(df_trn, 20000)

y_train, _ = split_vals(y_trn, 20000)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])

preds[:, 0], np.mean(preds[:, 0]), y_valid[0]
preds.shape
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
m = RandomForestRegressor(n_jobs=-1, n_estimators=20)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
df_trn, y_trn, z = proc_df(df_raw, 'SalePrice')

X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)
set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
reset_rf_samples()
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, min_samples_leaf = 3, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_jobs=-1, n_estimators=40, min_samples_leaf = 3, max_features = 0.5, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
df_raw
set_rf_samples(50000)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 

                        max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
%time preds = np.stack([t.predict(X_valid) for t in m.estimators_])

np.mean(preds[:,0]), np.std(preds[:,0])
def get_preds(t): return t.predict(X_valid)

%time preds = np.stack(parallel_trees(m, get_preds))

np.mean(preds[:,0]), np.std(preds[:,0])
x = raw_valid.copy()

x['pred_std'] = np.std(preds, axis=0)

x['pred'] = np.mean(preds, axis=0)

x.Enclosure.value_counts().plot.barh();
flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']

enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()

enc_summ
enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]

enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));
enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', 

              alpha=0.6, xlim=(0,11));
raw_valid.ProductSize.value_counts().plot.barh();
flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']

summ = x[flds].groupby(flds[0]).mean()

summ
(summ.pred_std/summ.pred).sort_values(ascending=False)
fi = rf_feat_importance(m, df_trn); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): 

  return fi.plot('cols','imp','barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.005].cols; len(to_keep)

df_keep = df_trn[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)



m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 

                       max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_keep)

plot_fi(fi);
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat = 7, )

X_train, X_valid = split_vals(df_trn2, n_trn)



m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_trn2)

plot_fi(fi[:25]);
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method = 'average')

fig = plt.figure(figsize = (16, 10))

dendrogram = hc.dendrogram(z, labels = df_keep.columns, orientation = 'left', leaf_font_size = 16)

plt.show()
def get_oob(df):

    m = RandomForestRegressor(n_estimators = 40, n_jobs = -1, min_samples_leaf = 5, max_features = 0.6, oob_score = True)

    x, _ = split_vals(df, n_trn)

    m.fit(x, y_train)

    return m.oob_score_
get_oob(df_keep)
for c in ('saleYear', 'saleElapsed', 'Hydraulics_Flow', 'Grouser_Tracks', 'fiModelDesc', 'fiBaseModel', 'Coupler_System'):

    print(c, get_oob(df_keep.drop(c, axis=1)))
death_sentence = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']

get_oob(df_keep.drop(death_sentence, axis=1))
df_keep.drop(death_sentence, axis = 1, inplace = True)

X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
from pdpbox import pdp

from plotnine import *
set_rf_samples(50000)
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat = 7, )

X_train, X_valid = split_vals(df_trn2, n_trn)



m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
plot_fi(rf_feat_importance(m, df_trn2)[:10] );
df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha = 0.01, figsize = (10, 8));
x_all = get_sample(df_raw[df_raw.YearMade > 1930], 500)
ggplot(x_all, aes('YearMade', 'SalePrice')) + stat_smooth(se = True, method = 'loess')
x = get_sample(X_train[X_train.YearMade > 1930], 500)
def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(m, x, x.columns, feat)

    return pdp.pdp_plot(p, feat_name, plot_lines=True, 

                        cluster=clusters is not None, 

                        n_cluster_centers=clusters)
plot_pdp('YearMade');
plot_pdp('YearMade', clusters=5);
feats = ['saleElapsed', 'YearMade']

p = pdp.pdp_interact(m, x, x.columns, feats)

pdp.pdp_interact_plot(p, feats)
plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure');
df_raw.YearMade[df_raw.YearMade<1950] = 1950

df_keep['age'] = df_raw['age'] = df_raw.saleYear-df_raw.YearMade
X_train, X_valid = split_vals(df_keep, n_trn)



m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)

m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep));
from treeinterpreter import treeinterpreter as ti
df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)
row = X_valid.values[None, 0]; row
predictions, bias, contributions = ti.predict(m, row)
predictions[0], bias[0]
idxs = np.argsort(contributions[0])
[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]
contributions[0].sum()
df_ext = df_keep.copy()

df_ext['is_valid'] = 1

df_ext.is_valid[:n_trn] = 0

x, y, na= proc_df(df_ext, 'is_valid')
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(x, y)

m.oob_score_
fi = rf_feat_importance(m, x)

fi[:10]
feats = ['SalesID', 'saleElapsed', 'MachineID']

(X_train[feats]/1000).describe()
(X_train[feats]/1000).describe()
x.drop(feats, axis = 1, inplace = True)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(x, y)

m.oob_score_
fi = rf_feat_importance(m, x)

fi[:10]
set_rf_samples(50000)
feats = ['SalesID', 'saleElapsed', 'MachineID', 'age', 'YearMade', 'saleDayofyear']
X_train, X_valid = split_vals(df_keep, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
for f in feats:

    df_subs = df_keep.drop(f, axis = 1)

    X_train, X_valid = split_vals(df_subs, n_trn)

    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

    m.fit(X_train, y_train)

    print(f)

    print_score(m)
reset_rf_samples()
df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis = 1)

X_train, X_valid = split_vals(df_subs, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
plot_fi(rf_feat_importance(m, X_train));
m = RandomForestRegressor(n_estimators=160, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)