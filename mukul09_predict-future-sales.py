import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from xgboost import plot_importance









import warnings

warnings.filterwarnings('ignore')
test_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv.gz', 

                        compression='gzip', header=0, sep=',', quotechar='"')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv.gz',

                    compression='gzip', header=0, sep=',', quotechar='"')

# Item count must be integer

sales['item_cnt_day'] = sales['item_cnt_day'].astype('int64')
print('test data :' + str( test_data.columns.tolist()))

print('item categories data :'+ str(item_categories.columns.tolist()))

print('items data :'+ str(items.columns.tolist()))

print('shops data :'+ str(shops.columns.tolist()))

print('sales data :'+ str(sales.columns.tolist()))
train_data = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
print(train_data.shape)

print(train_data.info())

train_data.head()

## Time period of the dataset

train_data['date'] = pd.to_datetime(train_data['date'], format="%d.%m.%Y")

print('The dataset starts from date: %s' % train_data['date'].min().date())

print('The dataset ends on date: %s' % train_data['date'].max().date())
test_data_shop_id = test_data['shop_id'].unique()

test_data_item_id = test_data['item_id'].unique()



# selecting only shops that are in test data

lk_train_data = train_data[train_data['shop_id'].isin(test_data_shop_id)]



# selecting only items that are in test data

lk_train_data = lk_train_data[lk_train_data['item_id'].isin(test_data_item_id)]
print(train_data.shape)

print(lk_train_data.shape)
train_data = train_data[train_data.item_price>0]
## we have shop id, item id and category id  that's why dropping shop name,

## item name and item category



monthly_train = lk_train_data[['date', 'date_block_num','shop_id','item_category_id',

                               'item_id', 'item_price', 'item_cnt_day']]
monthly_train.head()
## we have to find sales monthly

## therefore groupby date_block_num

monthly_train = monthly_train.sort_values('date').groupby(['date_block_num',

                                                          'shop_id',

                                                          'item_category_id',

                                                          'item_id'],

                                                         as_index = False)



monthly_train = monthly_train.agg({'item_price':['sum','mean'],'item_cnt_day':

                                  ['sum','mean','count']})



## Rename features

monthly_train.columns = ['date_block_num','shop_id','item_category_id',

                        'item_id','item_price','item_price_mean',

                        'item_cnt','item_cnt_mean','total_transactions']
monthly_train.head()
shop_ids = monthly_train['shop_id'].unique()

item_ids = monthly_train['item_id'].unique()

empty_df=[]

for i in range(34):

    for shop in shop_ids:

        for item in item_ids:

            empty_df.append([i,shop,item])

empty_df = pd.DataFrame(empty_df,columns=['date_block_num','shop_id','item_id'])
# Merging the empty_df with monthly_train (set missings records as 0)

monthly_train = pd.merge(empty_df, monthly_train, 

                         on =['date_block_num','shop_id','item_id'], 

                         how ='left')

monthly_train.fillna(0, inplace = True)
monthly_train.head(5)
# Extract time based features.

monthly_train['year'] = monthly_train['date_block_num'].apply(lambda x: ((x//12) + 2013))

monthly_train['month'] = monthly_train['date_block_num'].apply(lambda x: (x % 12)+1)
monthly_train.head()
#Grouping data for Analysis



# to analyze data month wise

gp_month_mean = monthly_train.groupby(['month'], as_index=False)['item_cnt'].mean()

gp_month_sum = monthly_train.groupby(['month'], as_index=False)['item_cnt'].sum()

    

# to analyze data item category wise

gp_category_mean = monthly_train.groupby(['item_category_id'], as_index=False)['item_cnt'].mean()

gp_category_sum = monthly_train.groupby(['item_category_id'], as_index=False)['item_cnt'].sum()



# to analyze data shop wise

gp_shop_mean = monthly_train.groupby(['shop_id'], as_index=False)['item_cnt'].mean()

gp_shop_sum = monthly_train.groupby(['shop_id'], as_index=False)['item_cnt'].sum()
f, axes = plt.subplots(2, 1, figsize=(22, 10))

sns.lineplot(x="month", y="item_cnt", data=gp_month_mean, ax=axes[0]).set_title("Monthly mean")

sns.lineplot(x="month", y="item_cnt", data=gp_month_sum, ax=axes[1]).set_title("Monthly sum")

plt.show()
f, axes = plt.subplots(2, 1, figsize=(22, 10))

sns.barplot(x="item_category_id", y="item_cnt", data=gp_category_mean, ax=axes[0]).set_title("Monthly mean")

sns.barplot(x="item_category_id", y="item_cnt", data=gp_category_sum, ax=axes[1]).set_title("Monthly sum")

plt.show()
f, axes = plt.subplots(2, 1, figsize=(22, 10))

sns.barplot(x="shop_id", y="item_cnt", data=gp_shop_mean, ax=axes[0]).set_title("Monthly mean")

sns.barplot(x="shop_id", y="item_cnt", data=gp_shop_sum, ax=axes[1]).set_title("Monthly sum")

plt.show()
sns.jointplot(x="item_cnt", y="item_price", data=monthly_train, height=8)

plt.show()

sns.jointplot(x="item_cnt", y="total_transactions", data=monthly_train, size=8)

plt.show()
monthly_train = monthly_train[(monthly_train['item_cnt']>0)& (monthly_train['item_cnt']<200)]

monthly_train = monthly_train[monthly_train['item_price']<400000]
monthly_train['item_cnt_month'] = monthly_train.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(-1)
monthly_train['item_price_unit'] = monthly_train['item_price'] // monthly_train['item_cnt']

monthly_train['item_price_unit'].fillna(0, inplace=True)
gp_item_price = monthly_train.sort_values('date_block_num').groupby(['item_id'], as_index=False).agg({'item_price':[np.min, np.max]})



gp_item_price.columns = ['item_id', 'hist_min_item_price', 'hist_max_item_price']



monthly_train = pd.merge(monthly_train, gp_item_price, on='item_id', how='left')
monthly_train['price_increase'] = monthly_train['item_price'] - monthly_train['hist_min_item_price']

monthly_train['price_decrease'] = monthly_train['hist_max_item_price'] - monthly_train['item_price']

monthly_train.head()
train_set = monthly_train.query('date_block_num <28').copy()

validation_set = monthly_train.query('date_block_num >= 28 and date_block_num < 33').copy()

test_set = monthly_train.query('date_block_num == 33').copy()

train_set.dropna(subset=['item_cnt_month'], inplace=True)

validation_set.dropna(subset=['item_cnt_month'], inplace=True)



train_set.dropna(inplace=True)

validation_set.dropna(inplace=True)



print('Train set records:', train_set.shape[0])

print('Validation set records:', validation_set.shape[0])

print('Test set records:', test_set.shape[0])
train_set.info()
#shop mean encoding

gp_shop_mean = train_set.groupby(['shop_id']).agg({'item_cnt_month':['mean']})

gp_shop_mean.columns = ['shop_mean']

gp_shop_mean.reset_index(inplace=True)



# Item mean encoding.

gp_item_mean = train_set.groupby(['item_id']).agg({'item_cnt_month': ['mean']})

gp_item_mean.columns = ['item_mean']

gp_item_mean.reset_index(inplace=True)



# Shop with item mean encoding.

gp_shop_item_mean = train_set.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})

gp_shop_item_mean.columns = ['shop_item_mean']

gp_shop_item_mean.reset_index(inplace=True)



# Year mean encoding.

gp_year_mean = train_set.groupby(['year']).agg({'item_cnt_month': ['mean']})

gp_year_mean.columns = ['year_mean']

gp_year_mean.reset_index(inplace=True)



# Month mean encoding.

gp_month_mean = train_set.groupby(['month']).agg({'item_cnt_month': ['mean']})

gp_month_mean.columns = ['month_mean']

gp_month_mean.reset_index(inplace=True)

# Add meand encoding features to train set.

train_set = pd.merge(train_set, gp_shop_mean, on=['shop_id'], how='left')

train_set = pd.merge(train_set, gp_item_mean, on=['item_id'], how='left')

train_set = pd.merge(train_set, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')

train_set = pd.merge(train_set, gp_year_mean, on=['year'], how='left')

train_set = pd.merge(train_set, gp_month_mean, on=['month'], how='left')



# Add meand encoding features to validation set.

validation_set = pd.merge(validation_set, gp_shop_mean, on=['shop_id'], how='left')

validation_set = pd.merge(validation_set, gp_item_mean, on=['item_id'], how='left')

validation_set = pd.merge(validation_set, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')

validation_set = pd.merge(validation_set, gp_year_mean, on=['year'], how='left')

validation_set = pd.merge(validation_set, gp_month_mean, on=['month'], how='left')
train_set.info()
# Create train and validation sets and labels. 

## since we have the relation of date_block_num 

## with the shop_id and item_id, therefore we can delete it

X_train = train_set.drop(['item_cnt_month', 'date_block_num'], axis=1)

y_train = train_set['item_cnt_month'].astype(int)

X_validation = validation_set.drop(['item_cnt_month', 'date_block_num'], axis=1)

y_validation = validation_set['item_cnt_month'].astype(int)
train_set.head()
latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')

X_test = pd.merge(test_set, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])

X_test['year'] = 2015

X_test['month'] = 11

X_test.drop('item_cnt_month', axis=1, inplace=True)

X_test = X_test[X_train.columns]
sets = [X_train, X_validation, X_test]



# Replace missing values with the median of each shop.            

for dataset in sets:

    for shop_id in dataset['shop_id'].unique():

        for column in dataset.columns:

            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()

            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median

            

# Fill remaining missing values on test set with mean.

X_test.fillna(X_test.mean(), inplace=True)
X_test.head()
from xgboost import XGBRegressor

xgb_model = XGBRegressor(max_depth=2, 

                         n_estimators=500, 

                         min_child_weight=1,  

                         colsample_bytree=1, 

                         subsample=1, 

                         eta=0.05, 

                         seed=0)

xgb_model.fit(X_train, 

              y_train, 

              eval_metric="rmse", 

              eval_set=[(X_train, y_train), (X_validation, y_validation)], 

              verbose=20, 

              early_stopping_rounds=20)
plt.rcParams["figure.figsize"] = (15, 6)

plot_importance(xgb_model)

plt.show()
# We use only those features which are more correlated to item_cnt_month

# We got the below results using Correlation matrix

imp_features = ['shop_item_mean','item_mean','item_cnt','item_cnt_mean','total_transactions','item_price','month_mean']



xgb_train = X_train[imp_features]

xgb_val = X_validation[imp_features]

xgb_test = X_test[imp_features]



xgb_model.fit(xgb_train, 

              y_train, 

              eval_metric="rmse", 

              eval_set=[(xgb_train, y_train), (xgb_val, y_validation)], 

              verbose=20, 

              early_stopping_rounds=20)

xgb_train_pred = xgb_model.predict(xgb_train)

xgb_val_pred = xgb_model.predict(xgb_val)

xgb_test_pred = xgb_model.predict(xgb_test)



print('Train rmse:', np.sqrt(mean_squared_error(y_train, xgb_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(y_validation, xgb_val_pred)))
# Use only part of features on random forest.



rf_train = X_train[imp_features]

rf_val = X_validation[imp_features]

rf_test = X_test[imp_features]



rf_model = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0, n_jobs=-1)

rf_model.fit(rf_train, y_train)



rf_train_pred = rf_model.predict(rf_train)

rf_val_pred = rf_model.predict(rf_val)

rf_test_pred = rf_model.predict(rf_test)



print('Train rmse:', np.sqrt(mean_squared_error(y_train, rf_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(y_validation, rf_val_pred)))
# Use only part of features on Linear Regression.



lr_train = X_train[imp_features]

lr_val = X_validation[imp_features]

lr_test = X_test[imp_features]



lr_scaler = MinMaxScaler()

lr_scaler.fit(lr_train)

lr_train = lr_scaler.transform(lr_train)

lr_val = lr_scaler.transform(lr_val)

lr_test = lr_scaler.transform(lr_test)



lr_model = LinearRegression(n_jobs=-1)

lr_model.fit(lr_train, y_train)



lr_train_pred = lr_model.predict(lr_train)

lr_val_pred = lr_model.predict(lr_val)

lr_test_pred = lr_model.predict(lr_test)



print('Train rmse:', np.sqrt(mean_squared_error(y_train, lr_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(y_validation, lr_val_pred)))
# Dataset that will be the train set of the ensemble model.

first_level = pd.DataFrame(xgb_val_pred, columns=['xgbboost'])

first_level['random_forest'] = rf_val_pred

first_level['linear_regression'] = lr_val_pred

first_level['label'] = y_validation.values

first_level.head()
# Dataset that will be the test set of the ensemble model.

first_level_test = pd.DataFrame(xgb_test_pred, columns=['xgbboost'])

first_level_test['random_forest'] = rf_test_pred

first_level_test['linear_regression'] = lr_test_pred

first_level_test.head()
meta_model = XGBRegressor(n_jobs=-1)



# Drop label from dataset.

first_level.drop('label', axis=1, inplace=True)

meta_model.fit(first_level, y_validation)



ensemble_pred = meta_model.predict(first_level)

final_predictions = meta_model.predict(first_level_test)
meta_model.get_params
print('Train rmse:', np.sqrt(mean_squared_error(ensemble_pred, y_validation)))