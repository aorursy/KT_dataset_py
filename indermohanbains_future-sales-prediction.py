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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import PolynomialFeatures

from scipy.stats import rankdata

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from category_encoders import TargetEncoder

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

import re

from sklearn.model_selection import StratifiedKFold, KFold

from nltk.tokenize import word_tokenize

from nltk.tokenize import regexp_tokenize

sales_train = '../input/competitive-data-science-predict-future-sales/sales_train.csv'

test = '../input/competitive-data-science-predict-future-sales/test.csv'

sample_submission = '../input/competitive-data-science-predict-future-sales/sample_submission.csv'

item_categories = '../input/competitive-data-science-predict-future-sales/item_categories.csv'

items = '../input/competitive-data-science-predict-future-sales/items.csv'

shops = '../input/competitive-data-science-predict-future-sales/shops.csv'

from datetime import date



import gc

import pickle



import holidays

import sys

train_data = pd.read_csv (sales_train)

test_data = pd.read_csv (test)

items_data = pd.read_csv (items)

shops_data = pd.read_csv (shops)

item_cat_data = pd.read_csv (item_categories)

submission_file = pd.read_csv (sample_submission)

test_data = test_data.drop ('ID', axis = 1)

print (train_data.shape)
train_data = train_data [train_data.duplicated () == False]

train_data.shape
def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df



train_data = downcast_dtypes(train_data)

test_data = downcast_dtypes(test_data)

items_data = downcast_dtypes(items_data)

shops_data = downcast_dtypes(shops_data)

item_cat_data = downcast_dtypes(item_cat_data)
col_dsi = ['date_block_num', 'shop_id', 'item_id']

col_si = ['shop_id', 'item_id']

col_di = ['date_block_num', 'item_id']

col_ds = ['date_block_num', 'shop_id']

col_d = ['date_block_num']

col_s = ['shop_id']

col_i = ['item_id']

col_dsicat = ['date_block_num', 'shop_id', 'item_category_id']

col_dicati = ['date_block_num', 'item_category_id', 'item_id']

col_dp = ['date_block_num', 'price_bins']
top_itemscat = list (items_data.groupby ('item_category_id')['item_id'].count ().nlargest (15).index)
# Fix category

l_cat = list(item_cat_data.item_category_name)

for ind in range(0,1):

    l_cat[ind] = 'PC Headsets / Headphones'

for ind in range(1,8):

    l_cat[ind] = 'Access'

    l_cat[8] = 'Tickets (figure)'

    l_cat[9] = 'Delivery of goods'

for ind in range(10,18):

    l_cat[ind] = 'Consoles'

for ind in range(18,25):

    l_cat[ind] = 'Consoles Games'

    l_cat[25] = 'Accessories for games'

for ind in range(26,28):

    l_cat[ind] = 'phone games'

for ind in range(28,32):

    l_cat[ind] = 'CD games'

for ind in range(32,37):

    l_cat[ind] = 'Card'

for ind in range(37,43):

    l_cat[ind] = 'Movie'

for ind in range(43,55):

    l_cat[ind] = 'Books'

for ind in range(55,61):

    l_cat[ind] = 'Music'

for ind in range(61,73):

    l_cat[ind] = 'Gifts'

for ind in range(73,79):

    l_cat[ind] = 'Soft'

for ind in range(79,81):

    l_cat[ind] = 'Office'

for ind in range(81,83):

    l_cat[ind] = 'Clean'

    l_cat[83] = 'Elements of a food'

item_cat_data ['item_category_name'] = pd.DataFrame (l_cat)

#items_data ['item_category_name'] = LabelEncoder ().fit_transform (items_data ['item_category_name'])



items_data = items_data.merge (item_cat_data, on = 'item_category_id', how = 'left').drop (['item_name'], axis =1)
shops_data ['city'] = shops_data ['shop_name'].apply (lambda x : x.split (' ')[0])

shops_data ['center'] = shops_data ['shop_name'].apply (lambda x : x.split (' ')[1])

shops_data = shops_data.drop ('shop_name', axis = 1)
items_data ['item_category_name'] = LabelEncoder ().fit_transform (items_data ['item_category_name']).astype (np.int8)
shops_data ['city'] = LabelEncoder ().fit_transform (shops_data ['city']).astype (np.int8)

shops_data ['center'] = LabelEncoder ().fit_transform (shops_data ['center']).astype (np.int8)
train_data ['date'] = pd.to_datetime (train_data ['date'], format = '%d.%m.%Y')

train_data ['item_price'] = round (train_data ['item_price'],1)
rus_holidays = holidays.Russia ()

train_data ['is_holiday'] = train_data ['date'].apply (lambda x : 1 if x in rus_holidays else 0)

holiday_month = train_data [train_data.is_holiday == 1].groupby ('date_block_num')['date'].nunique ()



holidays_df = pd.DataFrame (holiday_month).reset_index ()

holidays_df.columns = ['date_block_num', 'holiday_count']



master = pd.DataFrame (index = list (range (0,35)), columns = ['holidays']).reset_index ()

master.columns = ['date_block_num', 'holidays']

master ['holidays'] = 0



holidays = master.merge (holidays_df, on = 'date_block_num', how = 'left').fillna (0).drop ('holidays', axis = 1)



holidays ['date_block_num'] = holidays ['date_block_num'].astype (np.int8)

holidays ['holiday_count'] = holidays ['holiday_count'].astype (np.int8)
print (train_data.loc [train_data ['item_price']<0,:].shape [0])

print (train_data.loc [train_data ['item_cnt_day']<0,:].shape [0])

print (train_data ['item_price'].min ())

print (train_data ['item_cnt_day'].min ())
fig = plt.figure (figsize = (15,5))



fig.add_subplot (1,2,1)

train_data[['item_cnt_day']].boxplot ()



fig.add_subplot (1,2,2)

train_data [['item_price']].boxplot ()



plt.tight_layout ()
train_data = train_data.loc [(train_data ['item_price'] >= 0) & (train_data ['item_price'] < 300000) & (train_data ['item_cnt_day'] >=0) & (train_data ['item_cnt_day'] < 2000), :]

train_data = train_data [['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]
# inactive test_data items

td_piv = train_data.pivot_table (index = 'item_id', columns = ['date_block_num'], values = 'item_cnt_day', aggfunc = 'sum').fillna (0)

last6 = td_piv.loc [:,td_piv.columns [27:33]]

inactive_items = list (last6[last6.iloc [:, :].sum (axis = 1) == 0].index)

active_items = list (td_piv [td_piv.index.isin (inactive_items) == False].index)

print ('active items {}'.format (len (active_items)))

print ('Inactive items {}'.format (len (inactive_items)))



inactive_test_items = list(set(test_data.loc [test_data ['item_id'].isin (inactive_items), 'item_id'].values))

print ('inactive_test_items {}'.format (len (inactive_test_items)))



active_test_items = list (set(test_data.loc [test_data ['item_id'].isin (inactive_test_items) == False, 'item_id'].values))

print ('active_test_items {}'.format (len (active_test_items)))



# new test_data items

new_items = list(set (test_data.item_id)-set(train_data.item_id))



# no forecast test_data items

no_forecast_items = inactive_test_items + new_items



print ('no_forecast_items : {}'.format (len (no_forecast_items)))
train_data ['revenue'] = train_data ['item_price'] * train_data ['item_cnt_day']
revenue = train_data.groupby (col_dsi)['revenue'].sum ().reset_index ()

revenue.columns = col_dsi + ['revenue']





target = train_data.groupby (col_dsi)['item_cnt_day'].sum ().reset_index ()

target.columns = col_dsi + ['target']



price = train_data.groupby (col_dsi).agg ({'item_price' : ['mean']}).reset_index ().fillna (0)

price.columns = col_dsi + ['price']



train_data = train_data [col_dsi].merge (target, on = col_dsi, how = 'left').fillna (0)



train_data ['target'] = train_data ['target'].astype (np.float16)



train_data = train_data.merge (price, on = col_dsi, how = 'left').fillna (0)



train_data = train_data.merge (revenue, on = col_dsi, how = 'left').fillna (0)



train_data = downcast_dtypes(train_data)
train_data ['date_block_num'] = train_data ['date_block_num'].astype (np.int8)

train_data ['shop_id'] = train_data ['shop_id'].astype (np.int8)

train_data ['price'] = train_data ['price'].astype (np.float16)
from itertools import product

grid = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train_data[train_data.date_block_num==i]

    grid.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

grid = pd.DataFrame(np.vstack(grid), columns=cols)

grid ['date_block_num'] = grid ['date_block_num'].astype(np.int8)

grid ['shop_id'] = grid ['shop_id'].astype(np.int8)

grid ['item_id'] = grid ['item_id'].astype(np.int16)

grid .sort_values(cols,inplace=True)

train_data = grid.merge (train_data, on = col_dsi, how = 'left').drop_duplicates (subset = col_dsi, keep = 'first').fillna (0)

train_data = train_data.sort_values (by = col_dsi, ascending = True)
test_data ['date_block_num'] = 34

test_data ['date_block_num'] = test_data ['date_block_num'].astype (np.int8)

test_data ['shop_id'] = test_data ['shop_id'].astype (np.int8)

test_data ['item_id'] = test_data ['item_id'].astype (np.int16)
data = pd.concat ([train_data, test_data], axis = 0).fillna (0)
def lags (train, lags, col):

    col_dsi = ['date_block_num', 'shop_id', 'item_id']

    df = train

    

    for i in lags:

        prev = train [col_dsi + [col]].copy ()

        

        prev ['date_block_num'] = prev ['date_block_num'] + i

        prev = prev.rename (columns = {col : 'prev_' + col + str (i)})

        df = df.merge (prev, on = col_dsi, how = 'left')

       

    return df.fillna (0)
lags_ = [1,2,3,12]   

data = lags (data, lags_, 'target')
lags_ = [1]   

data = lags (data, lags_, 'price')
target_mean = data.groupby ('item_id', as_index = False)['price'].max ()

target_mean.columns = ['item_id'] + ['item_pricemax']



data = pd.merge (data, target_mean, on = ['item_id'], how = 'left')



target_mean = data.groupby ('item_id', as_index = False)['price'].min ()

target_mean.columns = ['item_id'] + ['item_pricemin']



data = pd.merge (data, target_mean, on = ['item_id'], how = 'left')



data ['discount'] = data ['item_pricemax'] - data ['item_pricemin']





lags_ = [1]   

data = lags (data, lags_, 'discount')

data = data.drop ('discount', axis = 1)



data = data.drop (['item_pricemax','item_pricemin'], axis = 1)
lags_ = [1]   

data = lags (data, lags_, 'revenue')
data ['revenue'] = data ['revenue'].astype (np.float32)

data ['prev_revenue1'] = data ['prev_revenue1'].astype (np.float32)
target_mean = data.groupby (col_di, as_index = False)['target'].mean ()

target_mean.columns = col_di + ['target_meandi']



data = pd.merge (data, target_mean, on = col_di, how = 'left')





lags_ = [1]   

data = lags (data, lags_, 'target_meandi')

data = data.drop ('target_meandi', axis = 1)
target_mean = data.groupby (col_ds, as_index = False)['target'].mean ()

target_mean.columns = col_ds + ['target_meands']



data = pd.merge (data, target_mean, on = col_ds, how = 'left')



lags_ = [1]   

data = lags (data, lags_, 'target_meands')

data = data.drop ('target_meands', axis = 1)
target_mean = data.groupby (col_ds, as_index = False)['revenue'].mean ()

target_mean.columns = col_ds + ['revenue_means']



data = pd.merge (data, target_mean, on = col_ds, how = 'left')



lags_ = [1]   

data = lags (data, lags_, 'revenue_means')

data = data.drop ('revenue_means', axis = 1)



target_mean = data.groupby (col_s, as_index = False)['revenue'].mean ()

target_mean.columns = col_s + ['revenue_means_tot']



data = pd.merge (data, target_mean, on = col_s, how = 'left')



lags_ = [1]   

data = lags (data, lags_, 'revenue_means_tot')

data = data.drop ('revenue_means_tot', axis = 1)

data ['revenue_delta'] = data ['prev_revenue_means_tot1'] - data ['prev_revenue_means1']

data = data.drop (['prev_revenue_means_tot1','prev_revenue_means1'], axis = 1)
target_mean = data.groupby (col_d, as_index = False)['target'].mean ()

target_mean.columns = col_d + ['M_mean']



data = pd.merge (data, target_mean, on = col_d, how = 'left')



lags_ = [1,2,3,12]

data = lags (data, lags_, 'M_mean') 

data = data.drop ('M_mean', axis = 1)
data = data.merge (shops_data, on = 'shop_id', how = 'left').fillna (0)

data = data.merge (items_data, on = 'item_id', how = 'left').fillna (0)
target_mean = data.groupby (col_dsicat, as_index = False)['target'].mean ()

target_mean.columns = col_dsicat + ['target_meansicat']



data = pd.merge (data, target_mean, on = col_dsicat, how = 'left')



lags_ = [1]   

data = lags (data, lags_, 'target_meansicat')

data = data.drop ('target_meansicat', axis = 1)
data ['inactive_items'] = data ['item_id'].isin (inactive_items).replace ({False:0, True:1}).astype (np.int16)



data = data.merge (holidays, on = 'date_block_num', how = 'left')



bins = np.linspace (data.price.min (), data.price.max (),6)

labels = ['very_low', 'low', 'medium', 'high','very_high']



data ['price_bins'] = pd.cut(data ['price'],bins,labels = labels, include_lowest = True)

data ['price_bins'] = data ['price_bins'].astype ('O')



data ['price_bins'] = LabelEncoder ().fit_transform (data ['price_bins']).astype (np.int8)
data ['month'] = data ['date_block_num'].apply (lambda x : x%12+1).astype (np.int8)



data ['year'] = data ['date_block_num'].apply (lambda x : x//12).astype (np.int8)
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

data['days'] = data['month'].map(days)

data['days'] = data['days'].astype (np.float16)
req_data = data.copy ()
req_data = req_data.drop (['item_category_id','price', 'revenue', 'shop_id', 'item_id'], axis = 1).fillna (0)

req_data = req_data.loc [req_data.date_block_num > 11, :]


del data

gc.collect()

del grid

gc.collect()

del target_mean

gc.collect()

del shops_data

gc.collect()

del items_data

gc.collect()

del target

gc.collect()

del price

gc.collect()

del train_data

gc.collect ()



del item_cat_data

gc.collect ()

del holidays

gc.collect ()

del revenue

gc.collect ()
req_data.info ()
X = req_data.loc [(req_data.date_block_num<33),:].drop (['date_block_num', 'target'], axis = 1)



y = req_data.loc [(req_data.date_block_num<33),:]['target'].clip (0,20)



X_valid = req_data [req_data ['date_block_num']==33].drop (['date_block_num', 'target'], axis = 1)

y_valid = req_data [req_data ['date_block_num']==33]['target'].clip (0,20)



X_test = req_data [req_data ['date_block_num']==34].drop (['date_block_num', 'target'], axis = 1)
del req_data

gc.collect();
corrtest = X.copy ()

corrtest ['target'] = y

plt.figure (figsize = (15,10))

sns.heatmap (corrtest.corr (), annot = True, cmap = 'viridis', mask = corrtest.corr () < 0.2)
model = XGBRegressor(

    max_depth=8,

    n_estimators=150,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42, reg_lambda=1, gamma = 0)



model.fit(

    X, 

    y, 

    eval_metric="rmse", 

    eval_set=[(X, y), (X_valid, y_valid)], 

    verbose=True, 

    early_stopping_rounds = 6)
prediction1 = model.predict (X_valid).clip (0,20)

prediction2 = model.predict (X_test).clip (0,20)
Score = sqrt (mean_squared_error (prediction1, y_valid))

print (round (np.mean (Score),4))
df = pd.DataFrame (model.get_booster().get_score(importance_type='weight'), index =[0]).transpose ()

df.columns = ['features']
df.sort_values (by = 'features', ascending = True).plot.barh (figsize = (7,10))
plt.scatter (prediction1, y_valid)

plt.ylim ([0,25])

plt.xlim ([0,25])
correlation = np.corrcoef (prediction1, y_valid)[0,1]

round (correlation,2)
output = pd.DataFrame({'ID': submission_file.ID, 'item_cnt_month': prediction2})

output.to_csv('submission.csv', index=False)