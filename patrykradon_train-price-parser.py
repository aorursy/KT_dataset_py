# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id']

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

calendar = calendar.reset_index()

calendar['index'] = calendar['index']+1
train_prices = pd.merge(left=sell_prices[['id','sell_price','wm_yr_wk']], right=calendar[['wm_yr_wk','index']], on='wm_yr_wk', how='left')

train_prices = pd.pivot_table(train_prices, index='id', columns='index', values='sell_price')
train_prices = train_prices.diff(axis=1)

train_prices = train_prices.ffill(axis=1)

train_prices = train_prices.bfill(axis=1)
train_prices.to_csv('./train_prices.csv')