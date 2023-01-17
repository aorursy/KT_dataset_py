# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn import preprocessing
%matplotlib inline 
from itertools import product
from tqdm import tqdm_notebook
import gc
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
print(train.head(5))
print(items.head(5))
print(categories.head(5))
print(shops.head(5))
print(test.head(5))
print(submission.head(5))
print(train.shape)
print(items.shape)
print(categories.shape)
print(shops.shape)
print(test.shape)
print(submission.shape)
print(train.describe())
print(items.describe())
print(categories.describe())
print(shops.describe())
print(test.describe())
print(submission.describe())
print(sns.heatmap(train.corr(),annot=True,cmap='rainbow'))
print(sns.heatmap(test.corr(),annot=True,cmap='rainbow'))
print(train.info())
print(items.info())
print(categories.info())
print(shops.info())
print(test.info())
print(submission.info())
gf1 = train
gf1[['day','month','year']] = gf1.date.str.split(".",expand=True)
gf1 = gf1.astype({"month": float, "year": int, "day": int})
gf1 = gf1.dropna(how = 'any')
a = gf1[(gf1.year >= 2013) ]
Optional_name_day = a.groupby('day').item_cnt_day.sum()

days = Optional_name_day.index[:]
total_num_items_sold_day = Optional_name_day[Optional_name_day].index[:]

plt.bar(days, total_num_items_sold_day)
plt.ylabel('Num items')
plt.xlabel('day')
plt.title("Num_items vs Day")
plt.show()

Optional_name_month = a.groupby('month').item_cnt_day.sum()

Month = Optional_name_month.index[:]
total_num_items_sold_month = Optional_name_month[Optional_name_month].index[:]

plt.bar(Month, total_num_items_sold_month)
plt.ylabel('Num items')
plt.xlabel('Month')
plt.title("Num_items vs Month")
plt.show()

Optional_name_year = a.groupby('year').item_cnt_day.sum()

Year = Optional_name_year.index[:]
total_num_items_sold_year = Optional_name_year[Optional_name_year].index[:]

plt.bar(Year, total_num_items_sold_year)
plt.ylabel('Num items')
plt.xlabel('year')
plt.title("Num_items vs Year")
plt.show()
print(set(gf1.item_cnt_day))
gf1["Ing_Dev"] = gf1["item_price"] * gf1["item_cnt_day"]
b = gf1
Optional_name_day_ID = b.groupby('day').Ing_Dev.sum()

days_ID = Optional_name_day_ID.index[:]
total_num_items_sold_day_ID = Optional_name_day_ID[Optional_name_day_ID].index[:]

plt.bar(days_ID, total_num_items_sold_day_ID)
plt.ylabel('Ing_Dev')
plt.xlabel('day')
plt.title("Ing_Dev vs Day")
plt.show()

Optional_name_month_ID = b.groupby('month').Ing_Dev.sum()

month_ID = Optional_name_month_ID.index[:]
total_num_items_sold_month_ID = Optional_name_month_ID[Optional_name_month_ID].index[:]

plt.bar(month_ID, total_num_items_sold_month_ID)
plt.ylabel('Ing_Dev')
plt.xlabel('month')
plt.title("Ing_Dev vs Month")
plt.show()

Optional_name_year_ID = b.groupby('year').Ing_Dev.sum()

year_ID = Optional_name_year_ID.index[:]
total_num_items_sold_year_ID = Optional_name_year_ID[Optional_name_year_ID].index[:]

plt.bar(year_ID, total_num_items_sold_year_ID)
plt.ylabel('Ing_Dev')
plt.xlabel('year')
plt.title("Ing_Dev vs Year")
plt.show()
b = gf1
Optional_name_shop_id = b.groupby('shop_id').Ing_Dev.sum()

days_ID = Optional_name_shop_id.index[:]
total_num_items_sold_shop_id = Optional_name_shop_id[Optional_name_shop_id].index[:]

plt.bar(days_ID, total_num_items_sold_shop_id)
plt.ylabel('Ingresos_Shop')
plt.xlabel('shop_id')
plt.title("Ingresos por Tienda")
plt.show()
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=0.5)

plt.subplot2grid((2,3),(0,0),colspan = 3)
train['shop_id'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('TRAIN_SHOP_ID_SET - NORMALIZED')

plt.subplot2grid((2,3),(1,0),colspan = 3)
train['date_block_num'].value_counts(normalize=True).plot(kind='bar',alpha=0.7)
plt.title('TRAIN_MONTH_COUNT_SET - NORMALIZED')

plt.show
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=0.25)

plt.subplot2grid((3,1),(0,0))
train['item_id'].plot(kind = 'hist', alpha=0.5, color='red')
plt.title('TRAIN_ITEM_ID_HISTOGRAM')

plt.subplot2grid((3,1),(1,0))
train['item_price'].plot(kind = 'hist', alpha=0.5, color='green')
plt.title('TRAIN_ITEM_PRICE_HISTOGRAM')

plt.subplot2grid((3,1),(2,0))
train['item_cnt_day'].plot(kind = 'hist', alpha=0.5, color='yellow')
plt.title('TRAIN_ITEM_COUNT_DAY_HISTOGRAM')

plt.show
fig, ax = plt.subplots(figsize = (15, 10))
sns.boxplot(x="shop_id", y="item_price", data=train, ax = ax)
fig, ax = plt.subplots(figsize = (15, 10))
sns.boxplot(x="shop_id", y="item_cnt_day", data=train, ax = ax)
shops
train.loc[train['shop_id'] == 0, 'shop_id'] = 57
train.loc[train['shop_id'] == 1, 'shop_id'] = 58
train.loc[train['shop_id'] == 10, 'shop_id'] = 11
shop_train = train['shop_id'].nunique()
print(shop_train)
cities = shops['shop_name'].str.split(' ').map(lambda row: row[0])
cities.unique()
shops['city'] = shops['shop_name'].str.split(' ').map(lambda row: row[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
LE = preprocessing.LabelEncoder()
shops['city_label'] = LE.fit_transform(shops['city'])
shops.drop(['shop_name', 'city'], axis = 1, inplace = True)
shops.head()
items_train = train['item_id'].nunique()
print(shop_train)
print(categories['item_category_name'])
prin_categories = categories['item_category_name'].str.split('-')

categories['main_category_id'] = prin_categories.map(lambda row: row[0].strip())
categories['main_category_id'] = LE.fit_transform(categories['main_category_id'])
categories['sub_category_id'] = prin_categories.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
categories['sub_category_id'] = LE.fit_transform(categories['sub_category_id'])

categories.head()
train = pd.merge(train, items, on=['item_id'], how='left')
train = pd.merge(train, categories, on=['item_category_id'], how='left')
train.head()
train = pd.merge(train, shops, on=['shop_id'], how='left')
train.head()
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=0.5)

plt.subplot2grid((3,3),(0,0),colspan = 3)
train['city_label'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('TRAIN_city_label_SET - NORMALIZED')

plt.subplot2grid((3,3),(1,0),colspan = 3)
train['main_category_id'].value_counts(normalize=True).plot(kind='bar',alpha=0.7)
plt.title('TRAIN_main_category_id_SET - NORMALIZED')

plt.subplot2grid((3,3),(2,0),colspan = 3)
train['sub_category_id'].value_counts(normalize=True).plot(kind='bar',alpha=0.7)
plt.title('TRAIN_sub_category_id_SET - NORMALIZED')

plt.show
train['item_id'].value_counts(ascending=False)[:5]
print(train.loc[train['item_id']==20949])
print(train.loc[train['item_id']==5822])
print(train.loc[train['item_id']==17717])
print(train.loc[train['item_category_id']==71])
print(train.loc[train['item_category_id']==35])
print(train.loc[train['item_category_id']==79])
print(test.loc[test['item_id']==20949].head(4))
print(test.loc[test['item_id']==5822].head(4))
print(test.loc[test['item_id']==17717].head(4))
print(train['item_cnt_day'].sort_values(ascending=False)[:5])
print(train[train['item_cnt_day']==2169])
print(train[train['item_cnt_day']==1000])
print(train[train['item_cnt_day']==669])

print(items[items['item_id']==20949])
print(items[items['item_id']==9248])

print(train[train['item_id']==11373].head(5))
print(train[train['item_id']==20949].head(5))
print(train[train['item_id']==9248].head(5))

train[train['item_id']==11373].item_cnt_day.plot(kind = 'hist', alpha=0.7, color='green')
plt.title('Item_Price_Histogram')
plt.show()

train[train['item_id']==20949].item_cnt_day.plot(kind = 'hist', alpha=0.7, color='green')
plt.title('Item_Price_Histogram')
plt.show()

train[train['item_id']==9248].item_cnt_day.plot(kind = 'hist', alpha=0.7, color='green')
plt.title('Item_Price_Histogram')
plt.show()
print(train['item_price'].sort_values(ascending=False)[:5])
print(train[train['item_price']==49782])
print(train[train['item_id']==6066])
train[(train['item_price']<49782) | ((train['item_price'] > 50999) & (train['item_price']<307980))]
print(train['item_price'].sort_values()[:5])
print(train[train['item_price']==-1])
print(train[(train['item_id']==2973) & (train['shop_id'] == 32)])
train = train[(train['item_cnt_day'] < 110)]
train = train[(train['item_price']<49782) | ((train['item_price'] > 50999) & (train['item_price']<307980))]
Correccion = train[(train['item_id']==2973) & (train['shop_id'] == 32) & (train['date_block_num'] == 4 & (train['item_price'] > 0))].item_price.median()
train.loc[train['item_price']<0,'item_price'] = Correccion
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=0.25)

plt.subplot2grid((3,3),(0,0), colspan = 3)
test['shop_id'].value_counts(normalize=True).plot(kind='bar',alpha=0.7)
plt.title('Shop_ID_VALUE_TEST_SET (N)')

plt.subplot2grid((3,3),(1,0))
test['item_id'].plot(kind='hist', alpha=0.7)
plt.title('Item_Id_Histogram_Test_Set')

plt.show
shop_test = test['shop_id'].nunique()
print(shop_test)

shop_train_list = list(gf1['shop_id'].unique())
shop_test_list = list(test['shop_id'].unique())

ok = 0
if(set(shop_test_list).issubset(set(shop_train_list))):
    ok = 1
    
print(ok)
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
shop_test = test['shop_id'].nunique()
print(shop_test)
items_test = test['item_id'].nunique()
print(shop_test)

#todos los items del test están en el train?
#Confirmar:

items_train_list = list(train['item_id'].unique())
items_test_list = list(test['item_id'].unique())

ok = 0
if(set(items_test_list).issubset(set(items_train_list))):
    ok = 1
    
print(ok)
#Si está en la lista, 1. Sino, 0. Entonces NO están todos en la lista. 
#Si no estan, como los vamos a predecir? -> Miremos la categoría de esos items
len(set(items_test_list).difference(items_train_list))
categories_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))].item_category_id.unique()
categories.loc[~categories['item_category_id'].isin(categories_in_test)]
shops_in_jan = train.loc[train['date_block_num']==0, 'shop_id'].unique()
items_in_jan = train.loc[train['date_block_num']==0, 'item_id'].unique()
jan = list(product(*[shops_in_jan, items_in_jan, [0]]))
shops_in_feb = train.loc[train['date_block_num']==1, 'shop_id'].unique()
items_in_feb = train.loc[train['date_block_num']==1, 'item_id'].unique()
feb = list(product(*[shops_in_feb, items_in_feb, [1]]))
shops_in_mar = train.loc[train['date_block_num']==2, 'shop_id'].unique()
items_in_mar = train.loc[train['date_block_num']==2, 'item_id'].unique()
mar = list(product(*[shops_in_mar, items_in_mar, [2]]))
shops_in_abr = train.loc[train['date_block_num']==3, 'shop_id'].unique()
items_in_abr = train.loc[train['date_block_num']==3, 'item_id'].unique()
abr = list(product(*[shops_in_abr, items_in_abr, [3]]))
print(len(jan))
print(len(feb))
print(len(mar))
print(len(abr))
cartesian_test = []
cartesian_test.append(np.array(jan))
cartesian_test.append(np.array(feb))
cartesian_test.append(np.array(mar))
cartesian_test.append(np.array(abr))
cartesian_test
cartesian_test = np.vstack(cartesian_test)
cartesian_test_df = pd.DataFrame(cartesian_test, columns = ['shop_id', 'item_id', 'date_block_num'])
cartesian_test_df.head()
cartesian_test_df.shape
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols]   = df[int_cols].astype(np.int16)
    
    return df
months = train['date_block_num'].unique()
cartesian = []
for month in months:
    shops_in_month = train.loc[train['date_block_num']==month, 'shop_id'].unique()
    items_in_month = train.loc[train['date_block_num']==month, 'item_id'].unique()
    from itertools import product
    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
cartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
cartesian_df.shape
x = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
x.head()
x.shape
new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
del x
del cartesian_df
del cartesian
del cartesian_test
del cartesian_test_df
del feb
del jan
del items_test_list
del items_train_list
del train
new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)
new_train.head()
test.insert(loc=3, column='date_block_num', value=34)
test['item_cnt_month'] = 0
test.head()
new_train = new_train.append(test.drop('ID', axis = 1))
new_train = pd.merge(new_train, shops, on=['shop_id'], how='left')
new_train.head()
new_train = pd.merge(new_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')
new_train.head()
new_train = pd.merge(new_train, categories.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')
new_train.head()
def generate_lag(train, months, lag_column):
    for month in months:
        # Speed up by grabbing only the useful bits
        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]
        train_shift['date_block_num'] += month
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train
del items
del categories
del shops
del test
new_train = downcast_dtypes(new_train)
gc.collect()
%%time
new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_cnt_month')
%%time
group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')
new_train = generate_lag(new_train, [1,2,3,6,12], 'item_month_mean')
new_train.drop(['item_month_mean'], axis=1, inplace=True)
%%time
group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')
new_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')
new_train.drop(['shop_month_mean'], axis=1, inplace=True)
%%time
group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
new_train = generate_lag(new_train, [1, 2], 'shop_category_month_mean')
new_train.drop(['shop_category_month_mean'], axis=1, inplace=True)
%%time
group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')

new_train = generate_lag(new_train, [1], 'main_category_month_mean')
new_train.drop(['main_category_month_mean'], axis=1, inplace=True)
%%time
group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()
new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')

new_train = generate_lag(new_train, [1], 'sub_category_month_mean')
new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)
new_train.tail()
new_train['month'] = new_train['date_block_num'] % 12
holiday_dict = {
    0: 6,
    1: 3,
    2: 2,
    3: 8,
    4: 3,
    5: 3,
    6: 2,
    7: 8,
    8: 4,
    9: 8,
    10: 5,
    11: 4,
}
new_train['holidays_in_month'] = new_train['month'].map(holiday_dict)
new_train = new_train[new_train.date_block_num > 11]
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            df[col].fillna(0, inplace=True)         
    return df

new_train = fill_na(new_train)
def xgtrain():
    regressor = xgb.XGBRegressor(n_estimators = 5000,
                                 learning_rate = 0.01,
                                 max_depth = 10,
                                 subsample = 0.5,
                                 colsample_bytree = 0.5)
    
    regressor_ = regressor.fit(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 
                               new_train[new_train.date_block_num < 33]['item_cnt_month'].values, 
                               eval_metric = 'rmse', 
                               eval_set = [(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 
                                            new_train[new_train.date_block_num < 33]['item_cnt_month'].values), 
                                           (new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1).values, 
                                            new_train[new_train.date_block_num == 33]['item_cnt_month'].values)
                                          ], 
                               verbose=True,
                               early_stopping_rounds = 10,
                              )
    return regressor_

%%time
regressor_ = xgtrain()
predictions = regressor_.predict(new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis = 1).values)
from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27

cols = new_train.drop('item_cnt_month', axis = 1).columns
plt.barh(cols, regressor_.feature_importances_)
plt.show()
submission['item_cnt_month'] = predictions
submission.to_csv('submission_JuanOsorio26.csv', index=False)
from IPython.display import FileLinks
FileLinks('.')
