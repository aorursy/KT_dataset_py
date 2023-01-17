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
from sklearn.preprocessing import LabelEncoder
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
item_category = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
item = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shop = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
'''dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0, aggfunc='sum')
dataset.reset_index(inplace = True)
dataset.head()'''
'''test.head()'''
'''dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')
dataset.fillna(0,inplace = True)
dataset.head()'''
'''dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)
dataset.head()'''
'''dataset.shape'''
'''# X we will keep all columns execpt the last one 
X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)
# the last column is our label
y_train = dataset.values[:,-1:]

# for test we keep all the columns execpt the first one
X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

# lets have a look on the shape 
print(X_train.shape,y_train.shape,X_test.shape)'''
'''from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout'''
'''baseline_model = Sequential()
baseline_model.add(LSTM(units = 64,input_shape = (33,1)))
baseline_model.add(Dense(1))

baseline_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
baseline_model.summary()'''
'''baseline_model.fit(X_train,y_train,batch_size = 4096,epochs = 10)'''
'''# creating submission file 
submission_pfs = baseline_model.predict(X_test)
# creating dataframe with required columns 
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_pfs.ravel()})
# creating csv file from dataframe
submission.to_csv('sub_pfs.csv',index = False)'''
'''submission.head()'''
def basic_EDA(df):
    print("-------Sample data---------")
    print(df.head())
    print("-------Description---------")
    print(df.describe())
    print("-------Information---------")
    print(df.info())
    print("-------Columns---------")
    print(df.columns)
    print("-------Data Types---------")
    print(df.dtypes)
    print("-------NULL and NA Values---------")
    print(df.isnull().sum())
    print(df.isna().sum())
    print("-------Shape---------")
    print(df.shape)
basic_EDA(train)
train['date'] = pd.to_datetime(train['date'],format = '%d.%m.%Y')
train.head()
train.dtypes
boxplot = train.boxplot(column=['item_cnt_day'])
boxplot = train.boxplot(column=['item_price'])
train[train.item_price > 100000]
train[train.item_price < 0]
train[train.item_cnt_day >= 1000]
train[train.item_cnt_day < 0]
#Let's check if the neg item_cnt_day are return?
'''neg = train[train.item_cnt_day < 0]
neg.head()

pos = train[train.item_cnt_day > 0]
pos.head()'''
#A function to check if the -ve item_cnt_day are return or not, but this function takes along time to run
#need to optimise it in some way so that it can be run in a reasonable amount of time
#Untill that time we will remove the -ve and continue as such.
'''return_counter = 0
for ind in neg.index:
    for i in range(ind):
        if ((neg['shop_id'][ind] == pos['shop_id'][i]) and 
        (neg['item_id'][ind] == pos['item_id'][i]) and 
        (neg['item_cnt_day'][ind] + pos['item_cnt_day'][i] == 0)):
            return_counter = return_counter + 1
        
            
print(return_counter)'''
train = train[(train.item_price < 300000 ) & (train.item_price > 0) & (train.item_cnt_day < 1000) & (train.item_cnt_day > 0)]
train.reset_index(inplace = True)
train.head()
basic_EDA(shop)
shop["city"] = shop.shop_name.str.split(" ").map( lambda x: x[0] )
shop["type"] = shop.shop_name.str.split(" ").map( lambda x: x[1] )
shop.head()
shop['city'].unique()
shop['type'].unique()
shop["shop_type"] = LabelEncoder().fit_transform( shop.type )
shop["shop_city"] = LabelEncoder().fit_transform( shop.city )
shop.head()
shop = shop[['shop_id', 'shop_city', 'shop_type']]
shop.head()
basic_EDA(item_category)
print(item_category['item_category_name'].str.split('-').map(lambda x: x[0]))
#print(item_category['item_category_name'].str.split('-').map(lambda x: x[1]))
item_category
print(item_category['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip()))
item_category['item_type'] = item_category['item_category_name'].str.split('-').map(lambda x: x[0])
item_category['item_name'] = item_category['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_category.head()
item_category['item_type'].unique()
item_category['item_name'].unique()
item_category['item_type_code'] = LabelEncoder().fit_transform(item_category.item_type)
item_category['item_name_code'] = LabelEncoder().fit_transform(item_category.item_name)
item_category.head()
item_category = item_category[['item_category_id', 'item_type_code', 'item_name_code']]
item_category.head()
basic_EDA(item)
list(item.item_name.unique())
item.shape
def clean_text(item_name):
    item_name = item_name.lower()
    item_name = re.sub(r'[^\w\s]', '', item_name)
    item_name = re.sub(r'\d+', '', item_name)
    item_name = re.sub(' +', ' ', item_name)
    item_name = item_name.strip()
    return item_name
item['clean_item_name'] = item['item_name'].apply(clean_text)
item.head()
item['sub_name1'] = item['clean_item_name'].str.split(' ').map(lambda x: x[0])
item['sub_name2'] = item['clean_item_name'].str.split(' ').map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
'''
item['sub_name3'] = item['clean_item_name'].str.split(' ').map(lambda x: x[2].strip() if len(x) > 2 else x[1].strip() if len(x) > 1 else x[0].strip())
item['sub_name4'] = item['clean_item_name'].str.split(' ').map(lambda x: x[3].strip() if len(x) > 3 else x[2].strip() if len(x) > 2 else x[1].strip() if len(x) > 1 else x[0].strip())
item['sub_name5'] = item['clean_item_name'].str.split(' ').map(lambda x: x[4].strip() if len(x) > 4 else x[3].strip() if len(x) > 3 else x[2].strip() if len(x) > 2 else x[1].strip() if len(x) > 1 else x[0].strip())
'''
item.head()
item.sub_name1.unique().shape, item.sub_name2.unique().shape
item.item_id.unique().shape, item.item_category_id.unique().shape
item['sub_name_1'] = LabelEncoder().fit_transform(item.sub_name1)
item['sub_name_2'] = LabelEncoder().fit_transform(item.sub_name2)
'''item['sub_name_3'] = LabelEncoder().fit_transform(item.sub_name3)
item['sub_name_4'] = LabelEncoder().fit_transform(item.sub_name4)
item['sub_name_5'] = LabelEncoder().fit_transform(item.sub_name5)'''
item.head()
item = item[['item_id', 'item_category_id', 'sub_name_1', 'sub_name_2']]
item.head()
basic_EDA(test)
len(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))
len(set(test.shop_id) - set(test.shop_id).intersection(set(train.shop_id)))
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
matrix.head()
matrix.dtypes, matrix.shape
train['revenue'] = train['item_price'] *  train['item_cnt_day']
train.head()
cols = ['date_block_num','shop_id','item_id']
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
group.head()
group.dtypes, group.shape
boxplot = group.boxplot(column = ['item_cnt_month'])
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                            .astype(np.float32)
                            .fillna(0))
matrix.head(10)
matrix.dtypes, matrix.shape
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
test.head()
test.dtypes, test.shape
matrix = pd.concat([matrix, test], ignore_index = True, keys = cols, sort = False)
matrix.fillna(0, inplace = True)
matrix.head()
matrix = pd.merge(matrix, shop, on = 'shop_id', how = 'left')
matrix = pd.merge(matrix, item, on = 'item_id', how = 'left')
matrix = pd.merge(matrix, item_category, on = 'item_category_id', how = 'left')
matrix.head()
matrix.dtypes
matrix.drop(['ID'], axis=1, inplace=True)
matrix.head()
matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
matrix['shop_type'] = matrix['shop_type'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['sub_name_1'] = matrix['sub_name_1'].astype(np.int16)
matrix['sub_name_2'] = matrix['sub_name_2'].astype(np.int16)
'''matrix['sub_name_3'] = matrix['sub_name_3'].astype(np.int16)
matrix['sub_name_4'] = matrix['sub_name_4'].astype(np.int16)
matrix['sub_name_5'] = matrix['sub_name_5'].astype(np.int16)'''
matrix['item_type_code'] = matrix['item_type_code'].astype(np.int8)
matrix['item_name_code'] = matrix['item_name_code'].astype(np.int8)
matrix.dtypes
matrix.head()
matrix['month'] = matrix['date_block_num'] % 12
matrix.head()
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)
matrix.head()
matrix.dtypes
#item_cnt_month of previous month, previous 2, 3, 6, and 12 month back
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted[col+'_lag_'+str(i)] = shifted[col+'_lag_'+str(i)].astype(np.float16)
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
matrix.head()
#Average and lag by date_block_num
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
matrix.head(10)
'''#Average and Lag by month
group = matrix.groupby(['month']).agg({'item_cnt_month':['mean']})
group.columns = ['month_avg_item_cnt']
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on=['month'], how = 'left')
matrix['month_avg_item_cnt'] = matrix['month_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'month_avg_item_cnt')
matrix.head(10)'''
#Average and Lag by data_block_num and shop_id
group = matrix.groupby(['date_block_num','shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_avg_item_cnt')
matrix.head(10)
'''#Average and Lag by month and shop_id
group = matrix.groupby(['month','shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'shop_id'], how='left')
matrix['month_shop_avg_item_cnt'] = matrix['month_shop_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_shop_avg_item_cnt')
matrix.head(10)'''
#Average and Lag by data_block_num and item_id
group = matrix.groupby(['date_block_num','item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_item_avg_item_cnt')
matrix.head(10)
'''#Average and Lag by month and item_id
group = matrix.groupby(['month','item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'item_id'], how='left')
matrix['month_item_avg_item_cnt'] = matrix['month_item_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_item_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by month and shop_city
group = matrix.groupby(['month','shop_city']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_shop_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'shop_city'], how='left')
matrix['month_shop_city_avg_item_cnt'] = matrix['month_shop_city_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_shop_city_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by month and shop_type
group = matrix.groupby(['month','shop_type']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_shop_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'shop_type'], how='left')
matrix['month_shop_type_avg_item_cnt'] = matrix['month_shop_type_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_shop_type_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by date_block_num and shop_city
group = matrix.groupby(['date_block_num','shop_city']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_city'], how='left')
matrix['date_shop_city_avg_item_cnt'] = matrix['date_shop_city_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'date_shop_city_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by date_block_num and shop_type
group = matrix.groupby(['date_block_num','shop_type']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_city'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by month and item_category_id
group = matrix.groupby(['month','item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'item_category_id'], how='left')
matrix['month_cat_avg_item_cnt'] = matrix['month_cat_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_cat_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by month and item_type_code
group = matrix.groupby(['month','item_type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_item_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'item_type_code'], how='left')
matrix['month_item_type_avg_item_cnt'] = matrix['month_item_type_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_item_type_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by month and item_name_code
group = matrix.groupby(['month','item_name_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'month_item_name_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month', 'item_name_code'], how='left')
matrix['month_item_name_avg_item_cnt'] = matrix['month_item_name_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'month_item_name_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by data_block_num and item_category_id
group = matrix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by data_block_num and item_type_code
group = matrix.groupby(['date_block_num','item_type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_type_code'], how='left')
matrix['date_item_type_avg_item_cnt'] = matrix['date_item_type_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'date_item_type_avg_item_cnt')
matrix.head(10)'''
'''#Average and Lag by month and item_name_code
group = matrix.groupby(['date_block_num','item_name_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_name_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_name_code'], how='left')
matrix['date_item_name_avg_item_cnt'] = matrix['date_item_name_avg_item_cnt'].astype(np.float16)
#matrix = lag_feature(matrix, [1], 'date_item_name_avg_item_cnt')
matrix.head(10)'''
matrix.columns
matrix.info()
X_train = matrix[matrix.date_block_num <= 33].drop(['item_cnt_month'], axis=1)
Y_train = matrix[matrix.date_block_num <= 33]['item_cnt_month']
X_valid = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = matrix[matrix.date_block_num == 33]['item_cnt_month']
X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)
X_train.shape, X_test.shape
model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('submission.csv', index=False)
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
plot_features(model, (10,14))