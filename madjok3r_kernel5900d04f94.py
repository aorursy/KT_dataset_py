import numpy as np
import pandas as pd
from datetime import datetime, timedelta,date
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_category = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shop = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

train = sales_train.groupby(["item_id","shop_id","date_block_num"]).sum().reset_index()

train = train.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})

train = train[["item_id","shop_id","date_block_num","item_cnt_month"]]
train
test.head()
train.isnull().sum()
from plotnine import *

montly_item_based_sum = train.groupby(['date_block_num']).sum().reset_index()


ggplot(aes(x='date_block_num', y='item_cnt_month'), montly_item_based_sum) + geom_line()
from plotnine import *

montly_shop_based_sum = montly_shop_based_sum.groupby(['shop_id']).sum().reset_index()

montly_shop_based_sum

ggplot(aes(x='shop_id', y='item_cnt_month'), montly_shop_based_sum) + geom_line()
num_month = train['date_block_num'].max()
month_list=[i for i in range(num_month+1)]
shop = []
for i in range(num_month+1):
    shop.append(5)
item = []
for i in range(num_month+1):
    item.append(5037)
months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})
months_full
clean= pd.merge(sales_train, train, how='right', on=['shop_id','item_id','date_block_num'])
clean = clean.sort_values(by=['date_block_num'])
clean.fillna(0.00,inplace=True)
clean
from sklearn.model_selection import train_test_split

X_full = clean.copy()
X_test_full = clean.copy()

X_full.dropna(axis=0, subset=['item_cnt_month'], inplace=True)
y = X_full.item_cnt_month
X_full.drop(['item_cnt_month'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=0)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

clf.fit(X_train, y_train)

preds = clf.predict(X_valid)

print(f'Model test sonuçları: {clf.score(X_valid, y_valid)*100:.3f}%')
# Add nan-values in the end of 'item_cnt_month', since there isnt as many rows in 'item_cnt_month'as in pred
sample_submission['item_cnt_month'] = pd.Series(preds)
sample_submission.apply(lambda col: col.drop_duplicates().reset_index(drop=True))
#make .csv file
sample_submission.to_csv('my_submission.csv', index=False)
print("Sonuç kayıt edildi!")
if len(sample_submission) == len(test):
    print("Sonuçlar uyumlu ({} rows).".format(len(sample_submission)))
else:
    print("Sonuçlar uyumlu değil")
