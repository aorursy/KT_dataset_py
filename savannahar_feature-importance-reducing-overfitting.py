!pip install fastai==0.7.0
%load_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn import metrics

from IPython.display import display
PATH = '../input/'
df_raw = pd.read_csv(PATH + 'train.csv',low_memory=False)
df_test = pd.read_csv(PATH + 'test.csv',low_memory=False)
df_raw.head(5)
df_raw.describe()
#check the missing value

df_raw.isnull().values.any()
df_trn, y_trn, nas = proc_df(df_raw, 'target')
def split_vals(a,n): return a[:n], a[n:]

n_valid = 30

n_trn = len(df_trn)-n_valid

X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)

raw_train, raw_valid = split_vals(df_raw, n_trn)
from sklearn.metrics import roc_auc_score



def auc(x,y): return roc_auc_score(x, y)#x - y_true, y = y_score



def print_score(m):

    res = [auc(y_train, m.predict(X_train)), auc(y_valid, m.predict(X_valid)),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
#This is definately overfitting

m = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
#a better fit

m = RandomForestRegressor(n_estimators=1000, min_samples_leaf=25, max_features=0.6, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(m, df_trn, y_trn, cv=5, scoring='roc_auc')

scores
fi = rf_feat_importance(m, df_trn);
#top 30 features are

fi[:30]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.005].cols; 

len_tokeep = len(to_keep)
df_keep = df_trn[to_keep].copy()

X_train, X_valid = split_vals(df_keep, 250)
X_train.shape
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=True)

scores = cross_val_score(m, X_train, y_trn, cv=5, scoring='roc_auc')

scores
m.fit(X_train, y_trn)
df_keep = df_test[to_keep].copy()
df_keep.shape
y_preds = m.predict(df_keep)
y_preds
submission_rf = pd.read_csv(PATH + 'sample_submission.csv')
submission_rf['target'] = y_preds
submission_rf.to_csv('submission_0.005.csv', index=False)
to_keep = fi[fi.imp>0.001].cols; 

len(to_keep)
df_keep = df_trn[to_keep].copy()

X_train, X_valid = split_vals(df_keep, 250)
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=True)

scores = cross_val_score(m, X_train, y_trn, cv=5, scoring='roc_auc')

scores
m.fit(X_train, y_trn)

df_keep = df_test[to_keep].copy()

y_preds = m.predict(df_keep)

submission_rf['target'] = y_preds

submission_rf.to_csv('submission_0.001.csv', index=False)
to_keep = fi[fi.imp>0.005].cols; 

df_keep = df_trn[to_keep].copy()

len_tokeep = len(to_keep)
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)

plt.show()