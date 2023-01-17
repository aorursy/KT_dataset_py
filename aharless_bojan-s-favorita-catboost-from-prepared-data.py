MAX_PRED = 1000

MAX_ROUNDS = 330
from datetime import date, timedelta

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor

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
val_pred = []

test_pred = []

cate_vars = []

for i in range(16):

    print("=" * 50)

    print("Step %d" % (i+1))

    print("=" * 50)

    model = CatBoostRegressor(

        iterations=MAX_ROUNDS, learning_rate=0.4,

        depth=4  )

        

    model.fit(

        X_train, y_train[:, i],

        cat_features=cate_vars,

        logging_level='Silent'

             )

    

    val_pred.append(model.predict(X_val))

    test_pred.append(model.predict(X_test))
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

submission.to_csv('cat_whatever.csv', float_format='%.4f', index=None)