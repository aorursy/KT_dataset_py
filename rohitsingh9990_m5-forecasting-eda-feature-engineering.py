import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import cycle

pd.set_option('max_columns', 50)

plt.style.use('bmh')

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
INPUT_DIR_PATH = '../input/m5-forecasting-accuracy/'



sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv')

calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')

sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv')

submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv')
print(f'Shape of sell_prices_df is: {sell_prices_df.shape}')

print(f'Shape of calendar_df is: {calendar_df.shape}')

print(f'Shape of sales_train_validation_df is: {sales_train_validation_df.shape}')

print(f'Shape of submission_df is: {submission_df.shape}')
sell_prices_df.head()
calendar_df.head()
sales_train_validation_df.head()
submission_df.head()
spd_profile = ProfileReport(sell_prices_df, title='sell_prices_df Profiling Report', html={'style':{'full_width':True}})
spd_profile.to_file(output_file="spd_profile.html")

spd_profile.to_notebook_iframe()

cd_profile = ProfileReport(calendar_df, title='calendar_df Profiling Report', html={'style':{'full_width':True}})
cd_profile.to_file(output_file="cd_profile.html")

cd_profile.to_notebook_iframe()
# using minimal=True to avoid heavy computation

# stvd_profile = ProfileReport(sales_train_validation_df, title='sales_train_validation_df Profiling Report', html={'style':{'full_width':True}}, minimal=True)

# stvd_profile.to_file(output_file="stvd_profile.html")

# stvd_profile.to_notebook_iframe()
# selecting 10 random rows from dataframe

stvd10 = sales_train_validation_df.sample(n = 10)

d_cols = [c for c in stvd10.columns if 'd_' in c] # sales data columns

stvd10 = stvd10.set_index('id')[d_cols].T
plt.figure(figsize=(40, 40))

plt.subplots_adjust(top=1.2, hspace = 0.8)

for i,item_id in enumerate(list(stvd10.columns)):

    plt.subplot(5, 2, i + 1)

    stvd10[item_id].plot(figsize=(20, 12),

          title=f'{item_id} sales by "d" number',

          color=next(color_cycle))

    plt.grid(False)

cal = calendar_df[['d','date','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',

       'snap_CA', 'snap_TX', 'snap_WI']]
stvd_1 = sales_train_validation_df.set_index('id')[d_cols].T

# rename id column to 'd', to perform merge operation

stvd_1 = stvd_1.reset_index().rename(columns={'index': 'd'})

# merging df cal and sales_train_validation_df on 'd'

stvd_merged = stvd_1.merge(cal, how='left', validate='1:1')
stvd10.head()
# rename id column to 'd', to perform merge operation

stvd10 = stvd10.reset_index().rename(columns={'index': 'd'})

stvd10 = stvd10.merge(cal, how='left', validate='1:1')

stvd10_date = stvd10.set_index('date')
stvd10_date.head()
plt.figure(figsize=(40, 40))

plt.subplots_adjust(top=1.2, hspace = 0.8)

for i,item_id in enumerate(list(stvd10_date.columns[1:11])):

    plt.subplot(5, 2, i + 1)

    stvd10_date[item_id].plot(figsize=(20, 12),

          title=f'{item_id} sales by "d" number',

          color=next(color_cycle))

    plt.tight_layout()

    plt.grid(False)
last_thirty_day_avg_sales = sales_train_validation_df.set_index('id')[d_cols[-30:]].mean(axis=1).to_dict()

fcols = [f for f in submission_df.columns if 'F' in f]

for f in fcols:

    submission_df[f] = submission_df['id'].map(last_thirty_day_avg_sales).fillna(0)

    

submission_df.to_csv('submission.csv', index=False)
submission_df.head()
## TODO:

# 1. Analyze sales of items by item_types i.e `Hobbies`, `Household`, `Foods`.

# 2. Analyze store wise sale of an item datewise

# 3. Analyze weekly trends or may be monthly.

# 4. Create new features

# 5. Try different models