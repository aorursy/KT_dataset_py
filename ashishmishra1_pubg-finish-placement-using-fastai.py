%load_ext autoreload
%autoreload 2
!pip install fastai==0.7.0 --no-deps
!pip install torch==0.4.1
%matplotlib inline
from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
df_raw = pd.read_csv('../input/train_V2.csv')
train_cats(df_raw)
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/pubg-raw')
df_raw = pd.read_feather('tmp/pubg-raw')
df_test = pd.read_csv('../input/test_V2.csv')
train_cats(df_test)
Id = df_test.Id
#df_raw.info()
#df_raw.isnull().sum().sort_values(ascending=False)
df_raw['winPlacePerc'].fillna(0,inplace=True)
df_trn, y_trn, nas = proc_df(df_raw, 'winPlacePerc')
df_test,_,_ = proc_df(df_test,na_dict = nas)
Id = df_test.Id
def split_vals(a,n): return a[:n], a[n:]
n_valid = 90000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
set_rf_samples(500000)
#m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
#m.fit(X_train, y_train)
#print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df_trn);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:29])
to_keep = fi[fi.imp>0.005].cols; len(to_keep)
df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df_keep)
plot_fi(fi);
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
#m = RandomForestRegressor(n_estimators=160, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
#%time m.fit(X_train, y_train)
#print_score(m)
df_ktest = df_test[to_keep].copy()
ypred = m.predict(df_ktest)
submission = pd.DataFrame({ 'Id': Id,
                            'winPlacePerc': ypred})
submission.to_csv("submission.csv", index=False)
