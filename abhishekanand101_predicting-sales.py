# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

from datetime import datetime



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

shops.head()
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

items.head()
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

item_categories.head()
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

sales_train.head()
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

test.set_index('ID', inplace = True)

test.head()
sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

sample_submission.head()
sales_train['date_format'] = pd.to_datetime(sales_train['date'], format = '%d.%m.%Y')

sales_train.drop(['date'], axis=1, inplace = True)

sales_train['month'] = sales_train['date_format'].dt.month

sales_train['year'] = sales_train['date_format'].dt.year

sales_train['date'] = sales_train['date_format'].dt.day

sales_train.drop(['date_format'], axis=1, inplace = True)

sales_train.head()
sales_train = sales_train.merge(items, left_on = "item_id", right_on = "item_id")

sales_train.drop(['item_name'], axis=1, inplace = True)

sales_train.head()
def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df



sales_train = downcast_dtypes(sales_train)

print(sales_train.info())
#sales_train.profile_report()
sales_train = sales_train[sales_train['item_cnt_day'] >= 0]

sales_train = sales_train[sales_train['item_price'] > 0]
sales_train.shape
sales_train.nunique(axis=0)
sales_train.describe().apply(lambda s: s.apply(lambda x: format (x, '.2f')))
sales_train.columns
train = sales_train[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'month', 'year', 'item_category_id']]

train['item_cnt_day'].clip(lower = 0, upper = 20, inplace = True)

train.head()
train.describe().apply(lambda s: s.apply(lambda x: format (x, '.2f')))
month_wise_sales = train.groupby(['month'])['item_cnt_day'].sum()

month_wise_sales = month_wise_sales - month_wise_sales.min()
sns.barplot(x=month_wise_sales.index, y = month_wise_sales)
train = train.groupby(['date_block_num', 'shop_id', 'item_id']).sum()

#train.drop(['month', 'year', 'item_category_id'], axis=1, inplace = True)

train
train.reset_index(inplace = True)
train['concat_month'] = (train['date_block_num']-1).astype(str) + " " + train['shop_id'].astype(str) + " " + train['item_id'].astype(str)

train['concat_year'] = (train['date_block_num']-12).astype(str) + " " + train['shop_id'].astype(str) + " " + train['item_id'].astype(str)

train['concat_curr_date'] = (train['date_block_num']).astype(str) + " " + train['shop_id'].astype(str) + " " + train['item_id'].astype(str)

train.head()



test['concat_month'] = "33 " + test['shop_id'].astype(str) + " " + test['item_id'].astype(str)

test['concat_year'] = "22 " + test['shop_id'].astype(str) + " " + test['item_id'].astype(str)

test['concat_curr_date'] = "34 " + test['shop_id'].astype(str) + " " + test['item_id'].astype(str)

train_month = train.copy()

train_month.set_index("concat_curr_date", inplace = True)



train_year = train.copy()

train_year.set_index("concat_curr_date", inplace = True)
def get_prev_month_sales(val):

    try:

        output = train_month.loc[val, 'item_cnt_day']

    except:

        output = 0

    return output



def get_prev_year_sales(val):

    try:

        output = train_year.loc[val, 'item_cnt_day']

    except:

        output = 0

    return output

train['prev_month_sales'] = train['concat_month'].map(get_prev_month_sales)

train['prev_year_sales'] = train['concat_year'].map(get_prev_year_sales)



test['prev_month_sales'] = test['concat_month'].map(get_prev_month_sales)

test['prev_year_sales'] = test['concat_year'].map(get_prev_year_sales)
#train.drop(['month', 'year', 'item_category_id'], axis=1, inplace = True)

train.head(-5)
train.nunique(axis=0)
sns.barplot(x=train['date_block_num'], y = train['item_cnt_day'])
#train.profile_report()
train = train.merge(items, left_on = "item_id", right_on = "item_id", how = 'left')

test = test.merge(items, left_on = "item_id", right_on = "item_id", how = 'left')
train['item_cnt_day'].clip(lower = 0, upper = 20, inplace = True)
train.head()
train.drop(['concat_month', 'concat_year', 'concat_curr_date', 'item_name', 'year', 'item_category_id_x'], axis=1, inplace = True)

test.drop(['concat_month', 'concat_year', 'concat_curr_date', 'item_name'], axis=1, inplace = True)
y = train.item_cnt_day

X = train.drop(['item_cnt_day', 'month'], axis=1)

X.rename(columns = {'item_category_id_y' : 'item_category_id'}, inplace = True)

X.head()
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(train_X, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_X, val_y)], 

             verbose=False)
plt.plot(my_model.feature_importances_)
feat_importances = pd.Series(my_model.feature_importances_, index=X.columns)

feat_importances.nlargest(7).plot(kind='barh')
test['date_block_num'] = 34
test.head()
test = test[X.columns]
pred = my_model.predict(test)

pred
pred = pred.clip(0, 20)
# Save test predictions to file

output = pd.DataFrame({'ID': test.index,

                       'item_cnt_month': pred})

output.to_csv('submission.csv', index=False)
output.head()