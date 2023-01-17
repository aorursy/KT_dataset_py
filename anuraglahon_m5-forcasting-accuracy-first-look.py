# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sales_train=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sales_train.head()
#dept_id unique elements

sales_train['dept_id'].unique()
sales_train.groupby('state_id').mean()
sales_train.groupby('item_id').mean()
sales_train.groupby(['state_id','item_id']).mean()
sales_train.describe()
sales_train.T
sales_train.isna().sum()
calendar=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

calendar.head()
calendar.describe()
calendar.isna().sum()
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sell_prices.head()
sell_prices.shape,calendar.shape,sales_train.shape
sell_prices.store_id.value_counts()
#group by store_id

sell_prices.groupby('store_id').mean()
sell_prices.describe()
sell_prices.isna().sum()
sell_prices.hist(column='sell_price',by='store_id',figsize=(13,17))