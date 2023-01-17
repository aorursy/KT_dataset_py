# Importing the libraries
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
# Importing the dataset
df_raw = pd.read_csv('../input/train.csv',low_memory=False,
                        parse_dates=['YearBuilt','YearRemodAdd','YrSold'])
df_raw['SalePrice'] = np.log(df_raw['SalePrice'])
df_raw.head()
(df_raw.isnull().sum().sort_index()/len(df_raw))
add_datepart(df_raw,'YearBuilt')
add_datepart(df_raw,'YearRemodAdd')
add_datepart(df_raw,'YrSold')
train_cats(df_raw)
# os.makedirs('tmp', exist_ok=True)
# df_raw.to_feather('tmp/df-raw')
# df_raw.to_feather('tmp\df-raw')
# df_raw = pd.read_feather('tmp\df-raw')
df ,y ,nas = proc_df(df_raw,'SalePrice')
from sklearn.ensemble import RandomForestRegressor
m = RandomForestRegressor(n_jobs=-1)
m.fit(df,y)
m.score(df,y)
# Splitting the data into training and test set
def split_vals(a,n): return a[:n].copy(),a[n:].copy()
n_valid = 440
n_trn = len(df_raw)-n_valid
raw_train,raw_valid = split_vals(df_raw,n_trn)
X_train,X_valid = split_vals(df,n_trn)
y_train,y_valid = split_vals(y,n_trn)

X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
# Score
def rmse(x,y): return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train),y_train),rmse(m.predict(X_valid),y_valid),
          m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,'oob_score_'): res.append(m.oob_score_)
    print(res)
from sklearn.ensemble import RandomForestRegressor
m = RandomForestRegressor(n_jobs=-1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)
res = []
ntrees = [20,40,50,100,200,400,500,1000]
for ntree in ntrees:
    m = RandomForestRegressor(n_jobs=-1,oob_score=True,n_estimators=ntree)
    m.fit(X_train,y_train)
    print_score(m)
max_feature_options = ['auto',None,'sqrt','log2',.9,.2,.5]
for max in max_feature_options:
    m = RandomForestRegressor(n_jobs=-1,oob_score=True,n_estimators=40,max_features=max)
    m.fit(X_train,y_train)
    print_score(m)
min_samples = [1,2,3,4,5,6,7,8,9]
for min in min_samples:
    m = RandomForestRegressor(n_jobs=-1,oob_score=True,n_estimators=40,max_features=.5,min_samples_leaf=min)
    m.fit(X_train,y_train)
    print_score(m)
m = RandomForestRegressor(n_jobs=-1,oob_score=True,n_estimators=40,max_features=.5,min_samples_leaf=1)
m.fit(X_train,y_train)
print_score(m)
%time preds = np.stack([t.predict(X_valid) for t in m.estimators_])
np.mean(preds[:,0]), np.std(preds[:,0])
x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.GarageType.value_counts().plot.barh()
flds = ['GarageType', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('GarageType', as_index=False).mean()
enc_summ
enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('GarageType', 'SalePrice', 'barh', xlim=(10,13))
enc_summ.plot('GarageType', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(10,13));
x.GarageCars.value_counts().plot.barh()
flds = ['GarageCars','SalePrice','pred','pred_std']
summ = x[flds].groupby(flds[0]).mean()
summ
(summ.pred_std/summ.pred).sort_values(ascending=False)
df_trn,y_trn,nas = proc_df(df_raw,'SalePrice')
fi = rf_feat_importance(m, df_trn); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30])
to_keep = fi[fi.imp>.005].cols; len(to_keep)
df_keep = df_trn[to_keep]
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_jobs=-1,oob_score=True,n_estimators=40,max_features=.5,min_samples_leaf=1)
m.fit(X_train,y_train)
print_score(m)
fi = rf_feat_importance(m, df_keep)
plot_fi(fi)
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
test = pd.read_csv('../input/test.csv',low_memory=False,
                        parse_dates=['YearBuilt','YearRemodAdd','YrSold'])
test.dtypes
add_datepart(test,'YearBuilt')
add_datepart(test,'YearRemodAdd')
add_datepart(test,'YrSold')
train_cats(test)
test,_,nas = proc_df(test)
y_pred = m.predict(test[to_keep])
submission = pd.DataFrame({'SalePrice':y_pred},index=test['Id'])
submission.to_csv('submission.csv')