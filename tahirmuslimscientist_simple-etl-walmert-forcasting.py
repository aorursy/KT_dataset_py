import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
MainDir = '/kaggle/input/m5-forecasting-accuracy/'
Calander_df = pd.read_csv(MainDir + 'calendar.csv')

Calander_df
Calander_df.columns
Calander_df.info()
sales_df = pd.read_csv(MainDir + 'sell_prices.csv')

sales_df
sales_df.loc[:,['store_id','item_id','sell_price']].groupby(['store_id','item_id']).sell_price.sum()
sales_train_df = pd.read_csv(MainDir + 'sales_train_validation.csv')

sales_train_df
sales_train_df.iloc[:,4:].groupby(['store_id','state_id']).sum()
sales_train_df.columns
sales_train_df.iloc[:,0:6]
sales_train_df.item_id.value_counts()
sales_train_df.state_id.value_counts()
Sub_df = pd.read_csv(MainDir + 'sample_submission.csv')
Sub_df.columns
Sub_df