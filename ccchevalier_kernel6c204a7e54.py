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
frame = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

frame
frame['date'] = pd.to_datetime(frame['date'], format="%d.%m.%Y")

frame.head()
to_be_grouped = ['shop_id', 'item_id', 'date_block_num', 'item_price']
grouped = frame.groupby(to_be_grouped, as_index=False).sum()

grouped.head()
lags = [1, 3, 6, 9]

mas = [3, 6, 9]
def create_lag_features(base_df, variable, lags, agg_cols):

    for lag in lags:

        column_name = variable + "_lag_" + str(lag)

        base_df[column_name] = base_df[variable].shift(-lag)



    return base_df
lagged_df = create_lag_features(

    grouped, 

    'item_cnt_day', 

    lags, 

    to_be_grouped)

lagged_df.head()
lagged_df.tail()