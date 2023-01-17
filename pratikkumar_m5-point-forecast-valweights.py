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
from collections import defaultdict

import sys
sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
sell_prices = sell_prices.set_index(['item_id', 'store_id', 'wm_yr_wk'])
sell_prices
sell_prices.loc[('HOBBIES_1_001', 'CA_1', 11327)].values[0]
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'])
calendar
calendar = calendar.set_index('d')
calendar
calendar.loc['d_1968', 'wm_yr_wk']
sales_train_val = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sales_train_val
def show_progress(progress_str, n_cols=80):

    sys.stdout.write(progress_str + (" " * max(0, n_cols - len(progress_str))) + "\r")

    sys.stdout.flush()
weights_df_dict = defaultdict(list)

for i in range(len(sales_train_val)):

    show_progress(f"Processing row {i+1}/{len(sales_train_val)}")

    _item_id = sales_train_val.item_id.iloc[i]

    _store_id = sales_train_val.store_id.iloc[i]

    _weight = 0.0

    _n = 0

    _ssum = 0.0

    _sales = []

    _day_of_first_sale = 1914

    

    for _d in range(1886,1914):

        _day_sales = sales_train_val['d_'+str(_d)].iloc[i]

        _wk_id = calendar.loc['d_'+str(_d), 'wm_yr_wk']

        _sp = sell_prices.loc[(_item_id, _store_id, _wk_id)].values[0]

        _weight += _day_sales * _sp

        

    for _d in range(1,1914):

        _day_sales = sales_train_val['d_'+str(_d)].iloc[i]

        if _n == 0:

            if _day_sales > 0:

                _n += 1

                _day_of_first_sale = _d

        else:

            _n += 1

        if _n>=2:

            _ssum += (_day_sales - sales_train_val['d_'+str(_d-1)].iloc[i])**2

            _sales.append(_day_sales)

    _sales = np.array(_sales).astype(np.float64)

    weights_df_dict['item_id'].append(_item_id)

    weights_df_dict['store_id'].append(_store_id)

    weights_df_dict['day_of_first_sale'].append(_day_of_first_sale)

    weights_df_dict['weight'].append(_weight)

    _all_zeros = 0

    if _n == 0:

        K = 0

        _all_zeros = 1

    else:

        K = pow(float(_n-1)/(_ssum), 0.5)

    weights_df_dict['K'].append(K)

    weights_df_dict['all_zeros'].append(_all_zeros)

    weights_df_dict['sales_mean'].append(_sales.mean())

    weights_df_dict['sales_std'].append(_sales.std())

weights_df_dict['weight'] = np.array(weights_df_dict['weight']).astype(np.float64)

weights_df_dict['K'] = np.array(weights_df_dict['K']).astype(np.float64)

weights_df_dict['all_zeros'] = np.array(weights_df_dict['all_zeros']).astype(np.int8)

weights_df_dict['sales_mean'] = np.array(weights_df_dict['sales_mean']).astype(np.float64)

weights_df_dict['sales_std'] = np.array(weights_df_dict['sales_std']).astype(np.float64)

weights_df_dict['day_of_first_sale'] = np.array(weights_df_dict['day_of_first_sale']).astype(np.int16)

weights_df = pd.DataFrame(weights_df_dict)
weights_df.loc[weights_df.all_zeros == 1]
wsum = weights_df.weight.sum(); wsum
weights_df['weight'] = weights_df.weight / wsum
weights_df
weights_val_val = pd.read_csv('../input/m5-point-forecast-valweights-actual/weights_validation.csv')
weights_val = weights_val_val.set_index(['Agg_Level_1', 'Agg_Level_2'])
weights_val.loc[list(zip(weights_df.item_id.values, weights_df.store_id.values)), 'Weight']
_a = weights_df.weight.values
_b = weights_val.loc[list(zip(weights_df.item_id.values, weights_df.store_id.values)), 'Weight'].values
(_a - _b).min(),(_a - _b).max() 
weights_df.to_pickle('weights_df')
!ls -alrh