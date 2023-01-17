MAX_PRED=1000
from datetime import date, timedelta

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

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
params = {

    'num_leaves': 31,

    'objective': 'regression',

    'min_data_in_leaf': 200,

    'learning_rate': 0.02,

    'feature_fraction': 0.8,

    'bagging_fraction': 0.8,

    'bagging_freq': 2,

    'metric': 'l2',

    'num_threads': 4

}
MAX_ROUNDS = 3000

val_pred = []

test_pred = []

cate_vars = []

for i in range(16):

    print("=" * 50)

    print("Step %d" % (i+1))

    print("=" * 50)

    dtrain = lgb.Dataset(

        X_train, label=y_train[:, i],

        categorical_feature=cate_vars,

        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1

    )

    dval = lgb.Dataset(

        X_val, label=y_val[:, i], reference=dtrain,

        weight=items["perishable"] * 0.25 + 1,

        categorical_feature=cate_vars)

    bst = lgb.train(

        params, dtrain, num_boost_round=MAX_ROUNDS,

        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=500

    )

    print("\n".join(("%s: %.2f" % x) for x in sorted(

        zip(X_train.columns, bst.feature_importance("gain")),

        key=lambda x: x[1], reverse=True

    )))

    val_pred.append(bst.predict(

        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))

    test_pred.append(bst.predict(

        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
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

submission.to_csv('lgb_whatever.csv', float_format='%.4f', index=None)