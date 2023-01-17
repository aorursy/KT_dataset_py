# IMPORT

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline
# LOAD TRANSACTIONS AND TRAIN

df_transactions= pd.read_csv("../input/transactions.csv", parse_dates = ['date'], infer_datetime_format = True)

types = {'id': 'int64','item_nbr': 'int32','store_nbr': 'int8','unit_sales': 'float32','onpromotion': bool,}

df_train= pd.read_csv("../input/train.csv", parse_dates = ['date'], dtype = types, infer_datetime_format = True)
# GROUP BY DATE AND CALCULATE RATIO

df_us = df_train.groupby(by='date').agg({'unit_sales':'sum'})

df_tr = df_transactions.groupby(by='date').agg({'transactions':'sum'})

df_ratio = df_us.merge(df_tr, how='outer', left_index=True, right_index=True)

df_ratio['ratio'] = df_ratio['unit_sales'] / df_ratio['transactions']
# COMPARE 2 SATURDAY 4 YEARS APART

print(df_ratio.loc['2013-04-06']) #Saturday

print(df_ratio.loc['2017-04-08']) #Saturday
# PLOT

f, ax = plt.subplots(3, 1, figsize=(20, 21))

df_ratio['unit_sales'].plot(ax=ax[0], title='unit_sales')

df_ratio['transactions'].plot(ax=ax[1], title='transactions')

df_ratio['ratio'].plot(ax=ax[2], title='ratio', ylim=(0,15))