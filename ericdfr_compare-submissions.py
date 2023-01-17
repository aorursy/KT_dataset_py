# imports

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline
# load train, test and items

types = {'id': 'int64','item_nbr': 'int32','store_nbr': 'int8','unit_sales': 'float32','onpromotion': bool,}

df_train= pd.read_csv("../input/favorita-grocery-sales-forecasting/train.csv", parse_dates = ['date'], dtype = types, infer_datetime_format = True)



types = {'id': 'int64','item_nbr': 'int32','store_nbr': 'int8','onpromotion': bool,}

df_test = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv", parse_dates = ['date'], dtype = types, infer_datetime_format = True)



df_items= pd.read_csv("../input/favorita-grocery-sales-forecasting/items.csv")
# load submission files

config = {'lgbm-0.529': ('../input/lgbm-starter/lgb.csv', 'blue'), \

          'logma-0.529': ('../input/log-ma-with-special-days-lb-0-529/ma8dspdays.csv.gz', 'green'), \

          'ets-0.556': ('../input/time-series-ets-starter-lb-0-556/sub_ets_log.csv', 'orange'), \

          'meanlog-0.622': ('../input/one-line-mean-log-lb-0-622/meanlog.csv.gz', 'pink')

         }



subs = {}

colors = {}

for key, conf in config.items():

    subs[key] = df_test.merge(pd.read_csv(conf[0]), on='id')

    colors[key] = conf[1]
picked = [] # keep track of previous comparisons when random
_LAST_X_DAYS = 120 # define time frame to display for train data, if None : display all available data

_RANDOM = False # change to True to display random comparison

if _RANDOM:

    df_s = df_test.sample(1)

    _store, _item = df_s['store_nbr'].values[0], df_s['item_nbr'].values[0]

    picked.append((_store, _item))

else:

    _store, _item = 34, 1239871

    

df_train_c = df_train[(df_train.store_nbr == _store) & (df_train.item_nbr == _item)].filter(items=['date', 'unit_sales']).set_index('date')



u_dates = [d for d in df_train.date.unique() if pd.Timestamp(d) >= df_train_c.index.min()]

df_train_c = df_train_c.reindex(u_dates)

df_train_c.unit_sales = df_train_c.unit_sales.fillna(0.0)



f, ax = plt.subplots(2, 1, figsize=(20, 20))



if df_train_c.shape[0] > 0:

    if _LAST_X_DAYS == None:

        df_train_c.rename(columns={"unit_sales": "unit_sales_train"}).plot(ax=ax[0], color='black')

    else:

        df_train_c[-_LAST_X_DAYS:].rename(columns={"unit_sales": "unit_sales_train"}).plot(ax=ax[0], color='black')



for k in subs.keys():

    df_plot = subs[k][(subs[k].store_nbr == _store) & (subs[k].item_nbr == _item)].filter(items=['date', 'unit_sales']).set_index('date').rename(columns={"unit_sales": "unit_sales_" + k})

    df_plot.plot(ax=ax[0], color=colors[k])

    df_plot.plot(ax=ax[1], color=colors[k])



plt.show()

_store, _item
# find most different store / item combos from 2 submissions according to nwrmsle

k1 = 'lgbm-0.529'

k2 = 'logma-0.529'



df_comp = df_test.merge(subs[k1], on='id').merge(subs[k2], on='id', suffixes=('_' + k1, '_' + k2)).merge(df_items, on='item_nbr').filter(items=['date', 'store_nbr', 'item_nbr', 'unit_sales_' + k1, 'unit_sales_' + k2, 'perishable'])

df_comp['perishable'] = df_comp['perishable'].apply(lambda x: 1.0 if x == 0 else 1.25)



def nwrmsle(x, k1, k2):

    return np.dot(np.square(np.log1p(x['unit_sales_' + k1])-np.log1p(x['unit_sales_' + k2])), np.transpose(x.perishable))



df_nwrmsle = df_comp.groupby(by=['store_nbr', 'item_nbr']).apply(nwrmsle, k1, k2)

df_nwrmsle = df_nwrmsle.reset_index()

df_nwrmsle.columns = ['store_nbr', 'item_nbr', 'nwrmsle']



df_nwrmsle.sort_values(by='nwrmsle', ascending=False).head(20)