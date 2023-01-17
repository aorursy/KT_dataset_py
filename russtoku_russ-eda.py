import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Where the competition data ends up after adding the hawaiiml data source
FILE_DIR = '../input/hawaiiml-data'
print(os.listdir(FILE_DIR))
train = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1')
train.head()
# basic pd info() doesn't reveal much about data values
train.info()
for column in ['invoice_id', 'customer_id', 'stock_id', 'date']:
    print('%-12s  %8.2d' % (column, len(train[column].unique())))
train[['invoice_id', 'customer_id']].groupby('customer_id')['invoice_id'].count().reset_index()
train['date'].sort_values().unique()[[0,1,2,-2,-1]]
train[['customer_id','country']].groupby('customer_id')['country'].unique()
train[train['customer_id'] == 7]['country'].unique()
train[['customer_id','invoice_id','country']][((train['customer_id'] == 7) & (train['country'] == 'eire'))].groupby('invoice_id').count()
train[['customer_id','invoice_id','country']][(train['customer_id'] == 7)].groupby('invoice_id').count()
train[['customer_id','invoice_id','date','time','stock_id','country']][((train['customer_id'] == 7) & (train['invoice_id'] == 7))].sort_values('stock_id')
