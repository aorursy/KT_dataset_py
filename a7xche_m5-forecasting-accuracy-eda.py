import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px





sns.set(style='darkgrid')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/m5-forecasting-accuracy"



sell_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))

sales_train_evaluation = pd.read_csv(os.path.join(path, "sales_train_evaluation.csv"))

calendar = pd.read_csv(os.path.join(path, "calendar.csv"))

sales_train_validation = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))
sell_prices.info()
sell_prices.isnull().sum()
sell_prices.head()
sell_prices["store_id"].unique()
sell_prices_item = sell_prices[["store_id", "item_id"]].drop_duplicates()

sell_prices_item.groupby("item_id").count().query("store_id != 10")
calendar.info()
calendar.isnull().sum()
calendar.head()
calendar.tail()
calendar.query("snap_CA == 1")
sales_train_validation.head()
# This code is from https://www.kaggle.com/tpmeli/visual-guide-3-m5-baselines-eda-sarima



d_cols = ['d_' + str(i + 1) for i in range(1913)]



tidy_df = pd.melt(frame = sales_train_validation, 

                  id_vars = ['id', 'item_id', 'cat_id', 'store_id'],

                  var_name = 'd',

                  value_vars = d_cols,

                  value_name = 'sales')

new_ids = tidy_df['id'] + '_' + tidy_df['d']

tidy_df['id'] = new_ids

group = sales_train_validation.groupby(['state_id','store_id','cat_id','dept_id'],as_index=False)['item_id'].count().dropna()

group['USA'] = 'United States of America'

group.rename(columns={'state_id':'State','store_id':'Store','cat_id':'Category','dept_id':'Department','item_id':'Count'},inplace=True)

fig = px.treemap(group, path=['USA', 'State', 'Store', 'Category', 'Department'], values='Count',

                  color='Count',

                  color_continuous_scale= px.colors.sequential.Sunset,

                  title='Walmart: Distribution of items')

fig.update_layout(template='seaborn')

fig.show()