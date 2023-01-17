# fastai 0.7.2

!pip install git+https://github.com/fastai/fastai@e85667cfae2e6873b1bb026195b5d09a74dfcff9

%load_ext autoreload

%autoreload 2



%matplotlib inline
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics
PATH = "../input/"



def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)



import os

import json

from pandas.io.json import json_normalize



def load_df(csv_path, nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    types = {

        "fullVisitorId": "str", # readme says it should be str

        "channelGrouping": "str",

        "date": "str",

        "socialEngagementType": "str",

        "visitId": "int32",

        "visitNumber": "int8",

        "visitStartTime": "int32",

    }

    

    df = pd.read_csv(csv_path, 

                     converters={column: json.loads for column in JSON_COLUMNS}, 

                     dtype=types,

                     nrows=nrows)

    

    for column in JSON_COLUMNS:

        column_as_df = json_normalize(df[column])

        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df
def split_vals(a,n): 

    return a[:n].copy(), a[n:].copy()



def rmse(x,y):

    return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
# df_raw = pd.read_csv(f'{PATH}train_v2.csv', low_memory=False, nrows=500_000, dtype={'fullVisitorId': 'str'})
# display_all(df_raw.head().T)
# display_all(df_raw.describe(include="all"))
df_raw = load_df(f'{PATH}train_v2.csv', nrows=500_000)
display_all(df_raw.head().T)
display_all(df_raw.describe(include="all"))
const_cols = [c for c in df_raw.columns if df_raw[c].nunique(dropna=False)==1]

df_raw.drop(columns=const_cols + ["customDimensions", "hits"], inplace=True)
df_test = load_df(f'{PATH}test_v2.csv')
display_all(df_test.head().T)
list((set(df_raw.columns).difference(set(df_test.columns))))
df_raw.drop(columns=['trafficSource.campaignCode'], inplace=True)
df_test.drop(columns=const_cols + ["customDimensions", "hits"], inplace=True)
df_raw['totals.transactionRevenue'].fillna(0, inplace=True)

df_raw['totals.transactionRevenue'] = df_raw['totals.transactionRevenue'].astype(float)

df_raw['totals.transactionRevenue'] = np.log1p(df_raw['totals.transactionRevenue'])
df_raw["visitStartTime"] = pd.to_datetime(df_raw["visitStartTime"], infer_datetime_format=True, unit="s")

df_raw["date"] = pd.to_datetime(df_raw["date"], infer_datetime_format=True, format="%Y%m%d")

add_datepart(df_raw, 'date')

add_datepart(df_raw, 'visitStartTime')
df_raw['totals.totalTransactionRevenue'].fillna(0, inplace=True)

df_raw['totals.totalTransactionRevenue'] = df_raw['totals.totalTransactionRevenue'].astype('int32')

df_raw['totals.transactions'].fillna(0, inplace=True)

df_raw['totals.transactions'] = df_raw['totals.transactions'].astype('int8')
df_test['totals.transactionRevenue'].fillna(0, inplace=True)

df_test['totals.transactionRevenue'] = df_test['totals.transactionRevenue'].astype(float)

df_test['totals.transactionRevenue'] = np.log1p(df_test['totals.transactionRevenue'])
df_test["visitStartTime"] = pd.to_datetime(df_test["visitStartTime"], infer_datetime_format=True, unit="s")

df_test["date"] = pd.to_datetime(df_test["date"], infer_datetime_format=True, format="%Y%m%d")

add_datepart(df_test, 'date')

add_datepart(df_test, 'visitStartTime')
display_all(df_raw.head().T)
train_cats(df_raw)

train_cats(df_test)
os.makedirs('tmp', exist_ok=True)

# df_raw.to_feather('tmp/ga-raw')

# df_test.to_feather('tmp/ga-test')
# df_raw = pd.read_feather('tmp/ga-raw')

# df_test = pd.read_feather('tmp/ga-test')
# display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
df_trn, y_trn, nas = proc_df(df_raw, 'totals.transactionRevenue')
display_all(df_trn.head().T)
m = RandomForestRegressor(n_jobs=-1)

m.fit(df_trn, y_trn)

m.score(df_trn, y_trn)
train_required_ratio = 0.8

n_trn = int(len(df_trn) * train_required_ratio)



X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)

X_train.shape, X_valid.shape
m = RandomForestRegressor(n_estimators=10, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])

preds[:,0], np.mean(preds[:,0]), y_valid[0]

preds.shape
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
set_rf_samples(50_000)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
df_trn, y_trn, nas = proc_df(df_raw, 'totals.transactionRevenue')

train_required_ratio = 0.8

n_trn = int(len(df_trn) * train_required_ratio)



X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)



X_test, y_test, _ = proc_df(df_test, 'totals.transactionRevenue', nas)

X_train.shape, X_valid.shape, X_test.shape
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)
# rmse(m.predict(X_train), y_train), rmse(m.predict(X_test), y_test), m.score(X_train, y_train), m.score(X_test, y_test), m.oob_score_

print_score(m)
m.feature_importances_
X_train.columns[22]
predictions = m.predict(X_test)
set_rf_samples(50000)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
raw_train, raw_valid = split_vals(df_raw, n_trn)
x = raw_valid.copy()

x['pred_std'] = np.std(preds, axis=0)

x['pred'] = np.mean(preds, axis=0)

x['totals.transactions'].value_counts().plot.barh();
flds = ['totals.transactions', 'totals.transactionRevenue', 'pred', 'pred_std']

tr_summ = x[flds].groupby('totals.transactions', as_index=False).mean()

tr_summ = tr_summ[~pd.isnull(tr_summ['totals.transactionRevenue'])]

tr_summ.plot('totals.transactions', 'totals.transactionRevenue', 'barh', xlim=(0,25));
tr_summ.plot('totals.transactions', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,25));
fi = rf_feat_importance(m, df_trn); fi[:30]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
fi[fi.imp>0.0001]
to_keep = fi[fi.imp>0.0001].cols; len(to_keep)
df_keep = df_trn[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,

                          n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_keep)

plot_fi(fi);
df_trn2, y_trn, nas = proc_df(df_raw, 'totals.transactionRevenue', max_n_cat=7)

X_train, X_valid = split_vals(df_trn2, n_trn)



m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_trn2)

plot_fi(fi[:25]);
from scipy.cluster import hierarchy as hc
df_keep = df_trn[to_keep].copy()
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
for c in ('totals.transactions', 'totals.totalTransactionRevenue'):

    print(c, get_oob(df_keep.drop(c, axis=1)))
to_drop = ['totals.transactions']

get_oob(df_keep.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)

X_train, X_valid = split_vals(df_keep, n_trn)
np.save('tmp/keep_cols.npy', np.array(df_keep.columns))
keep_cols = np.load('tmp/keep_cols.npy')

df_keep = df_trn[keep_cols]
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
!pip install treeinterpreter
from treeinterpreter import treeinterpreter as ti
df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)
row = X_valid.values[None,0]; row
prediction, bias, contributions = ti.predict(m, row)
prediction[0], bias[0]
idxs = np.argsort(contributions[0])
# [o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]
df_ext = df_keep.copy()

df_ext['is_valid'] = 1

df_ext.is_valid[:n_trn] = 0

x, y, nas = proc_df(df_ext, 'is_valid')
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(x, y);

m.oob_score_
fi = rf_feat_importance(m, x); fi[:10]
feats=['totals.timeOnSite']
(X_train[feats]/1000).describe()
(X_valid[feats]/1000).describe()
x.drop(feats, axis=1, inplace=True)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(x, y);

m.oob_score_
fi = rf_feat_importance(m, x); fi[:10]
set_rf_samples(50000)
feats=['totals.timeOnSite', 'geoNetwork.country', 'totals.hits', 'totals.pageviews']
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
reset_rf_samples()
df_subs = df_keep.drop(['geoNetwork.country'], axis=1)

X_train, X_valid = split_vals(df_subs, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
plot_fi(rf_feat_importance(m, X_train));
np.save('tmp/subs_cols.npy', np.array(df_subs.columns))
m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
X_test = X_test.drop(set(df_raw.columns).difference(set(df_subs.columns)) - {'totals.transactionRevenue'}, axis=1);
predictions = m.predict(X_test)
df_actual_test = df_test.copy()
actual_predicted_revenue = predictions

df_actual_test["predicted"] = actual_predicted_revenue



df_actual_test = df_actual_test[["fullVisitorId" , "predicted"]]

df_actual_test["fullVisitorId"] = df_actual_test.fullVisitorId.astype('str')

df_actual_test["predicted"] = df_actual_test.predicted.astype(np.float)

df_actual_test.index = df_actual_test.fullVisitorId

df_actual_test = df_actual_test.drop("fullVisitorId",axis=1)
df_actual_test["predicted"].value_counts().iloc[:5]
df_submission_test = pd.read_csv(filepath_or_buffer="../input/sample_submission_v2.csv",index_col="fullVisitorId")

df_submission_test.shape
"test shape is :{} and submission shape is : {}".format(df_actual_test.shape , df_submission_test.shape)

final_df = df_actual_test.loc[df_submission_test.index,:]
final_df = final_df[~final_df.index.duplicated(keep='first')]

final_df = final_df.rename(index=str, columns={"predicted": "PredictedLogRevenue"})
final_df.PredictedLogRevenue.value_counts(bins=3)
final_df = final_df.fillna(0); final_df.iloc[262153]
final_df.to_csv("sub.csv")