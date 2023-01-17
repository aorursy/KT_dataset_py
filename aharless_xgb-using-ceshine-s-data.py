from datetime import date, timedelta



import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import xgboost as xgb
df_train = pd.read_csv(

    '../input/train.csv', usecols=[1, 2, 3, 4, 5],

    dtype={'onpromotion': bool},

    converters={'unit_sales': lambda u: np.log1p(

        float(u)) if float(u) > 0 else 0},

    parse_dates=["date"],

    skiprows=range(1, 66458909)  # 2016-01-01

)



df_test = pd.read_csv(

    "../input/test.csv", usecols=[0, 1, 2, 3, 4],

    dtype={'onpromotion': bool},

    parse_dates=["date"]  # , date_parser=parser

).set_index(

    ['store_nbr', 'item_nbr', 'date']

)



items = pd.read_csv(

    "../input/items.csv",

).set_index("item_nbr")
df_train.shape
df_train.head()
df_test.shape
df_test.head()
items.shape
items.head()
df_2017 = df_train[df_train.date.isin(

    pd.date_range("2017-05-31", periods=7 * 11))].copy()

del df_train
df_2017.shape
df_2017.head()
promo_2017_train = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(

        level=-1).fillna(False)

promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)

promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)

promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)

promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)

del promo_2017_test, promo_2017_train
promo_2017.shape
promo_2017.head()
promo_2017.columns
df_2017 = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(

        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)

df_2017.shape
df_2017.head()
items = items.reindex(df_2017.index.get_level_values(1))

items.head()
items.shape
# Return that portion of the data frame that corresponds to the time period

#   beginning "minus" days before "dt" and extending for "periods" days

def get_timespan(df, dt, minus, periods):

    return df[

        pd.date_range(dt - timedelta(days=minus), periods=periods)

    ]
def prepare_dataset(t2017, is_train=True):

    X = pd.DataFrame({  # Mean target for different retrospective timespans & total # promotions

        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,

        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,

        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,

        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values

    })

    for i in range(16):  # Promotions on future days

        X["promo_{}".format(i)] = promo_2017[

            t2017 + timedelta(days=i)].values.astype(np.uint8)

    if is_train:

        y = df_2017[  # Target values for future days

            pd.date_range(t2017, periods=16)

        ].values

        return X, y

    return X
print("Preparing dataset...")

t2017 = date(2017, 6, 21)

X_l, y_l = [], []

for i in range(4):

    delta = timedelta(days=7 * i)

    X_tmp, y_tmp = prepare_dataset(

        t2017 + delta

    )

    X_l.append(X_tmp)

    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)

y_train = np.concatenate(y_l, axis=0)

del X_l, y_l

X_val, y_val = prepare_dataset(date(2017, 7, 26))

X_test = prepare_dataset(date(2017, 8, 16), is_train=False)
X_train.shape
X_train.head()
y_train.shape
y_train
X_val.shape
X_val.head()
y_val.shape
y_val
X_test.shape
X_test.head()
xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

    'n_jobs': 5

}
## THE BLOCK BELOW CUTS THIS DOWN TO THE FIRST 3 DAYS

n_preds = 16

n_preds = 3  # COMMENT OUT THIS LINE TO RUN FOR ALL DAYS

y_val = y_val[:, :n_preds]

y_val.shape
MAX_ROUNDS = 1000

val_pred = []

test_pred = []

cate_vars = []

for i in range(n_preds):

    print("=" * 50)

    print("Step %d" % (i+1))

    print("=" * 50)

    y_train_ = y_train[:, i]

    y_val_ = y_val[:, i]

    dtrain = xgb.DMatrix( X_train,  y_train_,

                          weight=pd.concat([items["perishable"]] * 4) * 0.25 + 1  )

    dval = xgb.DMatrix( X_val,  y_val_, 

                        weight=items["perishable"] * 0.25 + 1  )

    dtest = xgb.DMatrix( X_test )

    evals = [(dtrain,'train'),(dval,'eval')]

    bst = xgb.train(

        xgb_params, dtrain, num_boost_round=MAX_ROUNDS,

        evals=evals, early_stopping_rounds=50, verbose_eval=50

    )



    val_pred.append(bst.predict(dval, ntree_limit=bst.best_ntree_limit))

    test_pred.append(bst.predict(dtest, ntree_limit=bst.best_ntree_limit))
weights=pd.concat([items["perishable"]]) * 0.25 + 1

print("Validation mse:", mean_squared_error(

    y_val, np.array(val_pred).transpose(), sample_weight=weights))
#print("Making submission...")

#y_test = np.array(test_pred).transpose()

#df_preds = pd.DataFrame(

#    y_test, index=df_2017.index,

#    columns=pd.date_range("2017-08-16", periods=16)

#).stack().to_frame("unit_sales")

#df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)



#submission = df_test[["id"]].join(df_preds, how="left").fillna(0)

#submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
#submission.to_csv('lgb.csv', float_format='%.4f', index=None)