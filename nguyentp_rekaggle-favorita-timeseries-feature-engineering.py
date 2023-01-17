# インタネットをOnになる
!pip install py7zr
import os
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

print(os.listdir("../input"))
print(pd.__version__)
print(np.__version__)
import py7zr

for f in ['test.csv.7z', 'items.csv.7z', 'stores.csv.7z', 'train.csv.7z']:
    zf = py7zr.SevenZipFile(os.path.join('..', 'input', f), mode='r')
    zf.extractall('../working')
print(os.listdir('../working/'))
df_train = pd.read_csv(
    '../working/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # from 2016-01-01
)

df_test = pd.read_csv(
    "../working/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "../working/items.csv",
).set_index("item_nbr")

stores = pd.read_csv(
    "../working/stores.csv",
).set_index("store_nbr")
df_train.head()
df_test.head()
items.head()
stores.head()
le = LabelEncoder()
items['family'] = le.fit_transform(items['family'].values)
stores['city'] = le.fit_transform(stores['city'].values)
stores['state'] = le.fit_transform(stores['state'].values)
stores['type'] = le.fit_transform(stores['type'].values)
items.head()
stores.head()
df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
del df_train
df_2017.head()
df_2017.shape
promo_2017_train = df_2017.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_train.shape
promo_2017_train.head()
promo_2017_train.columns.get_level_values(1)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_train.head()
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test.shape
promo_2017_test.head()
# align test (store, item) index as same as train index.
# some (store, item) will be missing.
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017_test.shape
promo_2017_test.head()
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train
promo_2017.shape
promo_2017.head()
df_2017 = df_2017.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
df_2017.shape
df_2017.head()
df_2017.columns = df_2017.columns.get_level_values(1)
df_2017.shape
df_2017.head()
items.shape, stores.shape
items.shape, stores.shape
# align items
items = items.reindex(df_2017.index.get_level_values(1))
stores = stores.reindex(df_2017.index.get_level_values(0))
items.shape, stores.shape
items.head()
stores.head()
df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()
df_2017_item.shape
df_2017_item.head()
promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()
promo_2017_item.shape
promo_2017_item.head()
df_2017_store_class = df_2017.reset_index()
df_2017_store_class.shape
df_2017_store_class.head()
# df and items now have same row index.
df_2017_store_class['class'] = items['class'].values
df_2017_store_class.head()
df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]
df_2017_store_class_index.shape
df_2017_store_class_index.head()
df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()
df_2017_store_class.shape
df_2017_store_class.head()
df_2017_promo_store_class = promo_2017.reset_index()
df_2017_promo_store_class.shape
df_2017_promo_store_class.head()
df_2017_promo_store_class['class'] = items['class'].values
df_2017_promo_store_class.head()
df_2017_promo_store_class_index = df_2017_promo_store_class[['class', 'store_nbr']]
df_2017_promo_store_class = df_2017_promo_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()
df_2017_promo_store_class.shape
df_2017_promo_store_class.head()
def get_timespan(df, dt, minus, periods, freq='D'):
    """ Back minus days, get n==periods dates, each period is freq (D) away.
        if dt=6/16, minus=5, period=3, fred=D
        then return 6/11, 6/12, 6/13.
    """
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]
t2017 = date(2017, 6, 14)
pd.date_range(t2017, periods=16)
dt = t2017 = date(2017, 6, 14)
minus = 5
periods = 3
freq = "D"
pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)
print("Preparing dataset...")
t2017 = date(2017, 6, 14)
num_days = 6
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    print(t2017, delta, t2017 + delta)
def prepare_dataset(df, promo_df, t2017, is_train=True, name_prefix=None):
    """
    args:
    ----
        df: sale data
        promo_df: promo data
        t2017: pivot date
    """
    
    # Promotion counts.
    # How many promotions in last 14 days.
    # How many promotions in next 3 days. 
    X = {
        "promo_14_2017": get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,
        "promo_3_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values, 
        "promo_7_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,
        "promo_14_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,
    }

    # Sale on promotion and Non-promotion days.
    for i in [3, 7, 14]:
        # get sale in last i days.
        # get promo flag in last i days. if date has promo, value is 1. else, value is 0.
        tmp1 = get_timespan(df, t2017, i, i)
        tmp2 = (get_timespan(promo_df, t2017, i, i) > 0) * 1

        # average sale on promo dates in last i=3, 7, ... days. if last 3 days, 2 has promo, then average sale of these 2 promo dates.
        X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values
        # most recent dates has more influences on average sale.
        # if i == 3, np.power(0.9, np.arange(i)[::-1]) == [0.81, 0.9 , 1.  ]
        X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        # average sale on NON-promo dates in last i=3, 7, ... days. if last 3 days, 2 has NO-promo, then average sale of these 2 dates.
        X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values
        # most recent dates has more influences on average sale.
        # if i == 3, np.power(0.9, np.arange(i)[::-1]) == [0.81, 0.9 , 1.  ]
        X['no_promo_mean_%s_decay' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        
    # Stats on sale values.
    for i in [3, 7, 14]:
        # Get sale in last i days.
        tmp = get_timespan(df, t2017, i, i)
        # descritive stats on sale.
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values
        # weighted mean, most recent contribute more on average.
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        # get diff of day_T vs. day_T-1 > get mean.
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values


    # using same stats on sale, but now shift back 1 more week.
    for i in [3, 7, 14]:
        tmp = get_timespan(df, t2017 + timedelta(days=-7), i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values


    for i in [7, 14]:
        # sale in last 3 days.
        tmp = get_timespan(df, t2017, i, i)
        # how many days has sales in last 3 day
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        # distance to last day has sale. if i == 3, sales = [0, 4, 3], then distance = 1
        # if sales = [0, 4, 0], then distance = 2, etc...
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        # distance to first day has sale.
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

        # promo in last 3 days
        tmp = get_timespan(promo_df, t2017, i, i)
        # how many days has promo
        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        # distance to last day has promo
        X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        # distance to first day has promo
        X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values


    # promo in next 16 days
    tmp = get_timespan(promo_df, t2017 + timedelta(days=16), 15, 15)
    # how many promo in next 16 days
    X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values
    # distance to last day has promo
    X['last_has_promo_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values
    # distance to first day has promo
    X['first_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values

    # get sale in day t-1, t-2, t-3, ...
    for i in range(1, 7):
        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()

    # get promo in date ... t-3, t-2, t-1, t, t1, t2, t3...
    for i in range(-7, 7):
        X["promo_{}".format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

    # for each day of week, for example wednesday
    # get previous 4/20 wednesdays sales, then take average
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    X = pd.DataFrame(X)

    if is_train:
        y = df[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X
tmp1 = get_timespan(df_2017, t2017, 3, 3).head()
tmp1.head()
tmp2 = (get_timespan(promo_2017, t2017, 3, 3) > 0) * 1
tmp2.head(50)
(tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).head()
np.power(0.9, np.arange(3))
np.power(0.9, np.arange(3)[::-1])
tmp = get_timespan(df_2017, t2017, 3, 3)
tmp.head()
tmp.head().diff(axis=1)#.mean(axis=1)
tmp.head().diff(axis=1).mean(axis=1)
tmp = get_timespan(df_2017, t2017, 3, 3)
tmp.head()
(tmp > 0).sum(axis=1).head()
((tmp > 0) * np.arange(3)).head()
((tmp > 0) * np.arange(3)).head().max(axis=1)
3 - ((tmp > 0) * np.arange(3)).head().max(axis=1)
np.arange(3, 0, -1)
((tmp > 0) * np.arange(3, 0, -1)).head()
((tmp > 0) * np.arange(3, 0, -1)).head().max(axis=1)
t2017
get_timespan(df_2017, t2017, 28-0, 4, freq='7D').head()
t2017 = date(2017, 6, 14)
minus = 28
periods = 4
pd.date_range(t2017 - timedelta(days=minus), periods=periods, freq="7D")
df_2017.head()
promo_2017.head()
print("Preparing dataset...")
t2017 = date(2017, 6, 14)
num_days = 5
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    print(t2017, delta, t2017 + delta)
    print("process sales...")
    X_tmp, y_tmp = prepare_dataset(df_2017, promo_2017, t2017 + delta)
    print(X_tmp.shape, y_tmp.shape)

    print("process items...")
    X_tmp2 = prepare_dataset(df_2017_item, promo_2017_item, t2017 + delta, is_train=False, name_prefix='item')
    print(X_tmp2.shape)
    X_tmp2.index = df_2017_item.index
    print(X_tmp2.shape)
    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)
    print(X_tmp2.shape)

    print("process store...")
    X_tmp3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, t2017 + delta, is_train=False, name_prefix='store_class')
    print(X_tmp3.shape)
    X_tmp3.index = df_2017_store_class.index
    print(X_tmp3.shape)
    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)
    print(X_tmp3.shape)

    print("append sale, item, store, item-context, store-context data.")
    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)
    X_tmp['date'] = t2017 + delta
    print(X_tmp.shape)
          
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp2
    gc.collect()
    print("-" * 30)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_train.shape, y_train.shape
X_train.head()
y_train[:3]
os.remove('../working/stores.csv')
import pickle
with open(os.path.join('..', 'working', 'X_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)
    
import pickle
with open(os.path.join('..', 'working', 'y_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)
    
X_val, y_val = prepare_dataset(df_2017, promo_2017, date(2017, 7, 26))

X_val2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 7, 26), is_train=False, name_prefix='item')
X_val2.index = df_2017_item.index
X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_val3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')
X_val3.index = df_2017_store_class.index
X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)
X_val['date'] = date(2017, 7, 26)
X_val.shape, y_val.shape
with open(os.path.join('..', 'working', 'X_val.pkl'), 'wb') as f:
    pickle.dump(X_val, f)
with open(os.path.join('..', 'working', 'y_val.pkl'), 'wb') as f:
    pickle.dump(y_val, f)
X_test = prepare_dataset(df_2017, promo_2017, date(2017, 8, 16), is_train=False)

X_test2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 8, 16), is_train=False, name_prefix='item')
X_test2.index = df_2017_item.index
X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_test3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 8, 16), is_train=False, name_prefix='store_class')
X_test3.index = df_2017_store_class.index
X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_test = pd.concat([X_test, X_test2, X_test3, items.reset_index(), stores.reset_index()], axis=1)
X_test.shape
del X_test2, X_val2, df_2017_item, promo_2017_item, df_2017_store_class, df_2017_promo_store_class, df_2017_store_class_index
gc.collect()
