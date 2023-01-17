import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline
# Parameters

forecast_range = 28 # days

input_dir = '/kaggle/input/m5-forecasting-accuracy'
# Helper functions

def plot_time_series(data, ax=None, show=False):

    """Plot time series data."""

    if ax is None:

        fig, ax = plt.subplots()

    for ind in data.index:

        ax.plot([int(col.split('_')[-1]) for col in data.columns], 

                data.loc[ind].values, '-', label=ind)

    ax.legend(loc='best')

    ax.set_xlabel('day number')

    ax.set_ylabel('items sold')

    if show:

        plt.show(block=False)



def plot_aggregation(data):

    """Make plots over two time periods."""

    plot_time_series(data)

    plot_time_series(data[data.columns[-3*28:]])
# Load data

print('Loading data...')

sell_price = pd.read_csv('%s/sell_prices.csv' % input_dir)

calendar = pd.read_csv('%s/calendar.csv' % input_dir)

train = pd.read_csv('%s/sales_train_validation.csv' % input_dir).set_index('id')

sample_sub = pd.read_csv('%s/sample_submission.csv' % input_dir)



# Get column groups

cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

ts_cols = [col for col in train.columns if col not in cat_cols]

ts_dict = {t: int(t[2:]) for t in ts_cols}



# Describe data

print('  unique forecasts: %i' % train.shape[0])

for col in cat_cols:

    print('   N_unique %s: %i' % (col, train[col].nunique()))

# 1. All products, all stores, all states (1 series)

all_sales = pd.DataFrame(train[ts_cols].sum(axis=0)).transpose()

all_sales.index = ['all']

plot_aggregation(all_sales)
# 2. All products by state (3 series)

state_sales = train.groupby('state_id')[ts_cols].sum(axis=0)

plot_aggregation(state_sales)
# 3. All products by store (10 series)

store_sales = train.groupby('store_id')[ts_cols].sum(axis=0)

plot_aggregation(store_sales)
# 4. All products by category (3 series)

cat_sales = train.groupby('cat_id')[ts_cols].sum(axis=0)

plot_aggregation(cat_sales)
# 5. All products by department (7 series)

dept_sales = train.groupby('dept_id')[ts_cols].sum(axis=0)

plot_aggregation(dept_sales)
# 6. All products by state and category (9 series)

state_cat_sales = train.groupby(['state_id', 'cat_id'])[ts_cols].sum(axis=0)

plot_aggregation(state_cat_sales)
# 7. All products by state and category (21 series)

state_dept_sales = train.groupby(['state_id', 'cat_id'])[ts_cols].sum(axis=0)

plot_aggregation(state_dept_sales)
# 8. All products by store and category (30 series)

store_cat_sales = train.groupby(['store_id', 'cat_id'])[ts_cols].sum(axis=0)

plot_aggregation(store_cat_sales)
# 9. All products by store and department (70 series)

store_dept_sales = train.groupby(['store_id', 'dept_id'])[ts_cols].sum(axis=0)

plot_aggregation(store_dept_sales)