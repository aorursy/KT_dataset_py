VAL_DATE = '2016-08-01'

START_DATE = '2014-01-01'

MAX_ROUNDS = 2000
from datetime import date, timedelta

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
indir = '../input/second-week-sales/'

indir2 = '../input/favorita-grocery-sales-forecasting/'
print(check_output(["ls", indir[:-1]]).decode("utf8"))

print(check_output(["ls", indir2[:-1]]).decode("utf8"))
train_in = pd.read_csv(

    indir+'train_2nd_week.csv',

    dtype={'onpromotion': bool},

    converters={'unit_sales': lambda u: np.log1p(

        float(u)) if float(u) > 0 else 0},

    parse_dates=["date"],

)

print(train_in.shape)

train_in.head()
test_in = pd.read_csv(

    indir2+"test.csv",

    dtype={'onpromotion': bool},

    parse_dates=["date"]  # , date_parser=parser

)

print(test_in.shape)

test_in.head()
new_items = pd.read_csv(indir+'new_items.csv')

print(new_items.shape)

nnew = new_items.shape[0]

new_items.head()
test_new = test_in.merge(new_items, on='item_nbr', how='inner')

print(test_new.shape)

test_new.head()
all_2nd_week_items = pd.read_csv( indir+'items_2nd_week.csv', parse_dates=['date'] )

all_2nd_week_items.head()
stores = pd.read_csv(indir2+"stores.csv")

stores.head()
store_nbrs = stores[['store_nbr']].copy()

store_nbrs['dummy'] = 1

all_2nd_week_items['dummy'] = 1

all_combos = all_2nd_week_items.merge(store_nbrs, on='dummy', how='outer').drop('dummy',axis=1)

print( all_2nd_week_items.shape, store_nbrs.shape, all_combos.shape )

all_combos.head()
print(train_in.shape)

train_in.head()
train = all_combos.merge(train_in, on=['date','item_nbr','store_nbr'], how='outer')

train['unit_sales'] = train.unit_sales.fillna(0)

print(train.shape)

train.head()
items = pd.read_csv(indir2+"items.csv")

items.head()
df = train.merge(items, on='item_nbr')

test = test_new.merge(items, on='item_nbr')

df.head()
test.head()
store_code = df[df.date<VAL_DATE].groupby("store_nbr").unit_sales.sum().rename('store')

df = df.join(store_code, on='store_nbr', how='left')

test = test.join(store_code, on='store_nbr', how='left')

df.head()
test.head()
family_code = df[df.date<VAL_DATE].groupby("family").unit_sales.sum().rename('fam')

df = df.join(family_code, on='family', how='left')

test = test.join(family_code, on='family', how='left')

df.head()
test.head()
class_code = df[df.date<VAL_DATE].groupby("class").unit_sales.sum().rename('cla')

df = df.join(class_code, on='class', how='left')

test = test.join(class_code, on='class', how='left')

df.head()
df['dow'] = df['date'].dt.dayofweek

test['dow'] = test['date'].dt.dayofweek

dow_code = df[df.date<VAL_DATE].groupby("dow").unit_sales.sum().rename('day')

df = df.join(dow_code, on='dow', how='left')

test = test.join(dow_code, on='dow', how='left')

df['promo'] = df.onpromotion.astype(float)

test['promo'] = test.onpromotion.astype(float)

df.head()
df_train = df.drop(['date','store_nbr','item_nbr','family','class','dow','onpromotion'],axis=1)

df_test = test.drop(['id','date','store_nbr','item_nbr','family','class','dow','onpromotion'],axis=1)

df_train['store_class'] = df_train['store']*df_train['cla']

df_test['store_class'] = df_test['store']*df_test['cla']

df_train['store_fam'] = df_train['store']*df_train['fam']

df_test['store_fam'] = df_test['store']*df_test['fam']

df_train['store_day'] = df_train['store']*df_train['day']

df_test['store_day'] = df_test['store']*df_test['day']

df_train['store_prom'] = df_train['store']*(df_train['promo'].fillna(0.5))

df_test['store_prom'] = df_test['store']*(df_test['promo'].fillna(0.5))

df_train['store_per'] = df_train['store']*df_train['perishable']

df_test['store_per'] = df_test['store']*df_test['perishable']

df_train['class_day'] = df_train['cla']*df_train['day']

df_test['class_day'] = df_test['cla']*df_test['day']

df_train['class_prom'] = df_train['cla']*(df_train['promo'].fillna(0.5))

df_test['class_prom'] = df_test['cla']*(df_test['promo'].fillna(0.5))

df_train['class_per'] = df_train['cla']*df_train['perishable']

df_test['class_per'] = df_test['cla']*df_test['perishable']

df_train['day_prom'] = df_train['day']*(df_train['promo'].fillna(0.5))

df_test['day_prom'] = df_test['day']*(df_test['promo'].fillna(0.5))

df_train['day_per'] = df_train['day']*df_train['perishable']

df_test['day_per'] = df_test['day']*df_test['perishable']

df_train['per_prom'] = df_train['perishable']*(df_train['promo'].fillna(0.5))

df_test['per_prom'] = df_test['perishable']*(df_test['promo'].fillna(0.5))

df_train['fam_day'] = df_train['fam']*df_train['day']

df_test['fam_day'] = df_test['fam']*df_test['day']

df_train['fam_prom'] = df_train['fam']*(df_train['promo'].fillna(0.5))

df_test['fam_prom'] = df_test['fam']*(df_test['promo'].fillna(0.5))

df_train['fam_per'] = df_train['fam']*df_train['perishable']

df_test['fam_per'] = df_test['fam']*df_test['perishable']

df_train.head()
df_test.head()
df_val = df_train[df.date>=VAL_DATE]

df_train = df_train[(df.date<VAL_DATE) & (df.date>START_DATE)]
X_train = df_train.drop(['unit_sales'],axis=1).fillna(0.5)

y_train = df_train.unit_sales.values

X_val = df_val.drop(['unit_sales'],axis=1).fillna(0.5)

y_val = df_val.unit_sales.values

X_test = df_test.fillna(0.5)

X_train.head()
dtrain = lgb.Dataset( X_train, label=y_train, weight=X_train.perishable * 0.25 + 1 )

dval = lgb.Dataset( X_val, label=y_val, weight=X_val.perishable * 0.25 + 1 )
params = {

    'num_leaves': 100,

    'objective': 'regression',

    'min_data_in_leaf': 100,

    'learning_rate': 0.03,

    'feature_fraction': 0.7,

    'bagging_fraction': 0.9,

    'bagging_freq': 3,

    'metric': 'l2',

    'lambda_l2': .3,

    'lambda_l1': .02,

    'max_depth': 9,

    'min_split_gain': 1e-3,

    'num_threads': 4

}
bst = lgb.train( params, dtrain, num_boost_round=MAX_ROUNDS,

        valid_sets=[dtrain, dval], early_stopping_rounds=80, verbose_eval=50)
val_pred = bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS)

test_pred = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)

mse = mean_squared_error( y_val, np.array(val_pred), sample_weight=X_val.perishable * 0.25 + 1 )

np.sqrt(mse)
pd.Series(y_val).describe()
np.expm1(y_val.mean())
pd.Series(val_pred).describe()
pd.Series(y_train).describe()
pd.Series(test_pred).describe()
np.expm1(test_pred.mean())
scaledown = { # to account for gradual introudction of new products and ramp-up in first week

    '2017-08-16': .0001,

    '2017-08-17': .0002,

    '2017-08-18': .0006,

    '2017-08-19': .0013,

    '2017-08-20': .0020,

    '2017-08-21': .0033,

    '2017-08-22': .0048,

    '2017-08-23': .0063,

    '2017-08-24': .0078,

    '2017-08-25': .0093,

    '2017-08-26': .0106,

    '2017-08-27': .0120,

    '2017-08-28': .0128,

    '2017-08-29': .0136,

    '2017-08-30': .0143,

    '2017-08-31': .0151

}
sub = test[['id','date']].copy()

dates = list( sub.date.dt.strftime('%Y-%m-%d').values )

scales = [scaledown[d] for d in dates]

preds = np.expm1(test_pred * np.array(scales))

sub['unit_sales'] = preds

sub.drop(['id'],axis=1).set_index('date').apply(np.log1p).reset_index().groupby('date').mean().apply(np.expm1)
sub.drop(['date'],axis=1,inplace=True)

print(sub.shape)

sub.head()
sub.tail()
sub.to_csv('new_items_sub.csv', float_format='%.6f', index=None)