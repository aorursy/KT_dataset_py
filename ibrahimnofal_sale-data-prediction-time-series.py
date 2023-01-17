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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_train=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

df_test=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

df_shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

df_items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

df_itm_cat=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test_shops = df_test.shop_id.unique()

df_train = df_train[df_train.shop_id.isin(test_shops)]

test_items = df_test.item_id.unique()

df_train = df_train[df_train.item_id.isin(test_items)]

df_train.head(4)
fig, ax = plt.subplots(figsize=(35,15))

ax.scatter(df_train["date"][:1000], df_train["item_id"][:1000]);
df_train.item_cnt_day.hist();
sale=df_train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()
sale.head(4)
from math import ceil 

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))

num_graph = 10

id_per_graph = ceil(sale.shop_id.max() / num_graph)

count = 0

for i in range(5):

    for j in range(2):

        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', 

                      data=sale[np.logical_and(count*id_per_graph <= sale['shop_id'], 

                                               sale['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])

        count += 1
df_train.shop_id.unique()
df_train.item_id.value_counts()
df_train.item_id.unique()
df_train.date.dtype
# Import data again but this time parse dates

train_df=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv',

                    low_memory=False,

                 parse_dates=["date"])
train_df.date.dtype
train_df.head()
# sort data by date column

train_df.sort_values(by=['date'],inplace=True,ascending=True)
train_df.head()
# make a compy of train data

df_tmp=train_df.copy()
df_tmp["saleYear"] = df_tmp.date.dt.year

df_tmp["saleMonth"] = df_tmp.date.dt.month

df_tmp["saleDay"] = df_tmp.date.dt.day

df_tmp["saleDayOfWeek"] = df_tmp.date.dt.dayofweek

df_tmp["saleDayOfYear"] = df_tmp.date.dt.dayofyear
df_tmp.head(4)
plt.figure(figsize=(10, 5))

sns.countplot(df_tmp.saleYear, palette = 'bone')

plt.title('Comparison of Males and Females', fontweight = 30)

plt.xlabel('Year')

plt.ylabel('Count');

# missing data , checking numeric numbers



for label,content in df_tmp.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label)
# Check for which non numeric columns have null values

for label, content in df_tmp.items():

    if not pd.api.types.is_numeric_dtype(content):

        if pd.isnull(content).sum():

            print(label)
df_tmp.isna().sum()/len(df_tmp)
df_tmp.tail(6)
df_tmp.saleYear.value_counts()
# Split data into training and validation

df_val = df_tmp[df_tmp.saleYear == 2015]

df_train = df_tmp[df_tmp.saleYear != 2015]



len(df_val), len(df_train)
# Split data into X & y

X_train, y_train = df_train.drop(["item_cnt_day","date"], axis=1), df_train.item_cnt_day

X_valid, y_valid = df_val.drop(["item_cnt_day","date"], axis=1), df_val.item_cnt_day



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
import xgboost as xgb

# Creating the Regressor Model

model = xgb.XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300,

    colsample_bytree=0.8,

    subsample=0.8,

    eta=0.3, 

    seed=42

)



# Fitting the Model

model.fit(

    X_train,

    y_train,

    eval_metric="rmse",

    eval_set=[(X_train, y_train), (X_valid, y_valid)],

    verbose=True,

    early_stopping_rounds=10

)


preds = model.predict(X_valid)
preds
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_valid, preds))

print("RMSE: %f" % (rmse))
# Format predictions into the same format Kaggle is after

df_preds = pd.DataFrame()

df_preds["ID"] = X_valid["item_id"]

df_preds["item_cnt_month"] = preds

df_preds
# Export prediction data

df_preds.to_csv("submession.csv", index=False)