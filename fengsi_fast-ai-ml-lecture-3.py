from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics
PATH = '../input/'
!ls {PATH}
types = {
    'id': 'int64',
    'item_nbr': 'int32',
    'store_nbr': 'int8',
    'unit_sales': 'float32',
    'onpromotion': 'object'
}
%%time
df_all = pd.read_csv(f'{PATH}train.csv', parse_dates=['date'], dtype=types, 
                     infer_datetime_format=True, skiprows=range(1,100000000))
df_all.onpromotion.fillna(False, inplace=True)
df_all.onpromotion = df_all.onpromotion.map({'False': False, 'True': True})
df_all.onpromotion = df_all.onpromotion.astype(bool)

os.makedirs('tmp', exist_ok=True)
%time df_all.to_feather('tmp/raw_groceries')
%time df_all.describe(include='all')
df_test = pd.read_csv(f'{PATH}test.csv',parse_dates=['date'], dtype=types, infer_datetime_format=True)

df_test.onpromotion.fillna(False, inplace=True)
df_test.onpromotion = df_all.onpromotion.map({'False': False, 'True': True})
df_test.onpromotion = df_all.onpromotion.astype(bool)
df_test.describe(include='all')
df_all.tail()
df_all.unit_sales = np.log1p(np.clip(df_all.unit_sales, 0, None))
%time add_datepart(df_all, 'date')
def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = len(df_test)
n_trn = len(df_all) - n_valid
train, valid = split_vals(df_all, n_trn)
train.shape, valid.shape
%%time
trn, y, nas = proc_df(train, 'unit_sales')
val, y_val, nas = proc_df(valid, 'unit_sales')
def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(x), y), rmse(m.predict(val), y_val),
          m.score(x, y), m.score(val, y_val)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
set_rf_samples(1000000)
%time x = np.array(trn, dtype=np.float32)
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=100, n_jobs=-1)
%time m.fit(x, y)
print_score(m)
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=10, n_jobs=-1)
%time m.fit(x, y)
print_score(m)
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=10, n_jobs=-1)
%time m.fit(x, y)
print_score(m)