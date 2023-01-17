MAX_PRED = 1000

MAX_ROUNDS = 100
from datetime import date, timedelta

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
indir = '../input/preparing-data-for-lgbm-or-something-else/'

indir2 = '../input/favorita-grocery-sales-forecasting/'
X_test = pd.read_csv(indir + 'X_test.csv')

X_val = pd.read_csv(indir + 'X_val.csv')

X_train = pd.read_csv(indir + 'X_train.csv')

y_train = np.array(pd.read_csv(indir + 'y_train.csv'))

y_val = np.array(pd.read_csv(indir + 'y_val.csv'))

stores_items = pd.read_csv(indir + 'stores_items.csv', index_col=['store_nbr','item_nbr'])

test_ids = pd.read_csv( indir + 'test_ids.csv',  parse_dates=['date']).set_index(

                        ['store_nbr', 'item_nbr', 'date'] )
items = pd.read_csv( indir2 + 'items.csv' ).set_index("item_nbr")

items = items.reindex( stores_items.index.get_level_values(1) )
param = {}

param['objective'] = 'reg:linear'

param['eta'] = 0.5

param['max_depth'] = 3

param['silent'] = 1

param['eval_metric'] = 'rmse'

param['min_child_weight'] = 4

param['subsample'] = 0.8

param['colsample_bytree'] = 0.7

param['seed'] = 137

plst = list(param.items())
val_pred = []

test_pred = []

cate_vars = []

dtest = xgb.DMatrix(X_test)

for i in range(16):

    print("=" * 50)

    print("Step %d" % (i+1))

    print("=" * 50)

    dtrain = xgb.DMatrix(

        X_train, label=y_train[:, i],

        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1

    )

    dval = xgb.DMatrix(

        X_val, label=y_val[:, i],

        weight=items["perishable"] * 0.25 + 1)

        

    watchlist = [ (dtrain,'train'), (dval, 'val') ]

    model = xgb.train(plst, dtrain, MAX_ROUNDS, watchlist, early_stopping_rounds=50, verbose_eval=50)

    

    val_pred.append(model.predict(dval))

    test_pred.append(model.predict(dtest))
n_public = 5 # Number of days in public test set

weights=pd.concat([items["perishable"]]) * 0.25 + 1

print("Unweighted validation mse: ", mean_squared_error(

    y_val, np.minimum( np.array(val_pred).transpose(), np.log1p(MAX_PRED) ) )   )

mse = mean_squared_error(

    y_val, np.minimum( np.array(val_pred).transpose(), np.log1p(MAX_PRED) ), 

    sample_weight=weights)

print("Full validation mse:       ", mse )

msepub = mean_squared_error(

    y_val[:,:n_public], 

    np.minimum( np.array(val_pred).transpose()[:,:n_public], np.log1p(MAX_PRED) ),

    sample_weight=weights)

print("'Public' validation mse:   ",  msepub )

msepriv = mean_squared_error(

    y_val[:,n_public:], 

    np.minimum( np.array(val_pred).transpose()[:,n_public:], np.log1p(MAX_PRED) ),

    sample_weight=weights)

print("'Private' validation mse:  ",  msepriv )

print('Validation NRMSWLE')

print( "  Full:    ", np.sqrt(mse) )

print( "  Public:  ", np.sqrt(msepub) )

print( "  Private: ", np.sqrt(msepriv) )
y_test = np.array(test_pred).transpose()

df_preds = pd.DataFrame(

    y_test, index=stores_items.index,

    columns=pd.date_range("2017-08-16", periods=16)

).stack().to_frame("unit_sales")

df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
submission = test_ids.join(df_preds, how="left").fillna(0)

submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, MAX_PRED)

submission.to_csv('xgb_whatever.csv', float_format='%.4f', index=None)