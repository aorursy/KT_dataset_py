# pip install xgboost --upgrade
# to check version 
import xgboost as xgb
xgb.__version__
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge , Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import tensorflow as tf
import time

# To ignore unwanted warnings
import warnings
warnings.filterwarnings('ignore')
df_sales_train0 = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
df_shops = pd.read_csv('../input/predict-future-sales-translated-dataset/shops_en.csv')
df_items = pd.read_csv('../input/predict-future-sales-translated-dataset/items_en.csv')
df_catog = pd.read_csv('../input/predict-future-sales-translated-dataset/item_categories_en.csv')
df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
display(df_sales_train0.head())
df_sales_train0.shape
df_sales_train0['date_block_num'].nunique()
df_sales_train0['shop_id'].nunique()
df_sales_train0['item_id'].nunique()
df_catog.head()
df_catog['item_category_name'].nunique()
df_items.head()
df_items['item_name'].nunique()
df_shops.head()
df_shops['shop_name'].nunique()
test.head()
df.head()
df_sales_train0.shape
#this figure before filtering train data by taking only item_id and shop_id that exists in the test data  

Z = df_sales_train0.groupby('date_block_num').agg({'item_cnt_day': sum}).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
sns.barplot(data=Z, x='date_block_num', y='item_cnt_day', ax = ax, palette="BrBG")
plt.title('Total Sales Per Month', fontsize=25)
plt.xlabel('Months', fontsize=25)
plt.ylabel('Sales', fontsize=25);

#also this figure before filtering train data by taking only item_id and shop_id that exists in the test data  

price_per_month = df_sales_train0.groupby('date_block_num').agg({'item_price': 'mean'}).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
sns.barplot(data=price_per_month, x='date_block_num', y='item_price', ax = ax, palette="BrBG")
plt.title('Average Items Price Per Month', fontsize=25)
plt.xlabel('Months', fontsize=25)
plt.ylabel('Price', fontsize=25);
sns.boxplot(df_sales_train0['item_price'])
sns.boxplot(df_sales_train0['item_cnt_day'])
#removing item_cnt_day bigger than 1001 and item_price bigger than 100000
df_sales_train0 = df_sales_train0[df_sales_train0['item_cnt_day'] < 1001]
df_sales_train0 = df_sales_train0[df_sales_train0['item_price'] < 100000]
from itertools import product
df_sales_train = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = df_sales_train0[df_sales_train0['date_block_num'] == i]
    df_sales_train.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique())), dtype='int16'))
    
df_sales_train = pd.DataFrame(np.vstack(df_sales_train), columns=cols)
df_sales_train0['revenue'] = df_sales_train0['item_price'] *  df_sales_train0['item_cnt_day']
agg = df_sales_train0.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
agg.columns = ['item_cnt_month']
agg.reset_index(inplace=True)
merge1 = pd.merge(df_sales_train, agg, on=cols, how='left')
test['date_block_num'] = 34
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
# Only shops that exist in test set.
merge1 = merge1[merge1['shop_id'].isin(test_shop_ids)]
# Only items that exist in test set.
merge1 = merge1[merge1['item_id'].isin(test_item_ids)]
merge1.reset_index(inplace=True, drop=True)
merge1 = pd.concat([merge1, test], ignore_index=True, sort=False, keys=['date_block_num','shop_id','item_id'])
merge1.fillna(0, inplace=True)
merge1 = pd.merge(merge1, df_shops, on=['shop_id'], how='left')
merge1 = pd.merge(merge1, df_items, on=['item_id'], how='left')
merge1 = pd.merge(merge1, df_catog, on=['item_category_id'], how='left')
merge1.loc[merge1.shop_id == 0, 'shop_id'] = 57
merge1.loc[merge1.shop_id == 1, 'shop_id'] = 58
merge1.loc[merge1.shop_id == 10, 'shop_id'] = 11
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
merge1 = lag_feature(merge1, [1,2,3,6,12], 'item_cnt_month')  # Create months lags
CPI_inflation = [0.97,0.56,0.34,0.51,0.66,0.42,0.82,0.14,0.21,0.57,0.57,0.50,
                 0.59,0.70,1.02,0.90,0.90,0.62,0.49,0.24,0.65,0.82,1.28,2.62,
                 3.85,2.22,1.21,0.46,0.35,0.19,0.80,0.35,0.57,0.74,0.75]
merge1['CPI_inflation'] = 0
for i, value in enumerate(CPI_inflation):
    merge1['CPI_inflation'][merge1['date_block_num'] == i ] = value

merge1 = lag_feature(merge1, [1,2,3,6,12], 'CPI_inflation')  # Creating lags for CPI_inflation
agg = merge1.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
agg.columns = [ 'date_avg_item_cnt' ]
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num'], how='left')
merge1 = lag_feature(merge1, [1], 'date_avg_item_cnt')
# merge1.drop(['date_avg_item_cnt'], axis=1, inplace=True)
agg = merge1.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
agg.columns = [ 'date_item_avg_item_cnt' ]
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num','item_id'], how='left')
merge1 = lag_feature(merge1, [1,2,3,6,12], 'date_item_avg_item_cnt')
# merge1.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
agg = merge1.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
agg.columns = [ 'date_shop_avg_item_cnt' ]
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num','shop_id'], how='left')
merge1 = lag_feature(merge1, [1,2,3,6,12], 'date_shop_avg_item_cnt')
# merge1.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

agg = merge1.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
agg.columns = [ 'date_cat_avg_item_cnt' ]
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num','item_category_id'], how='left')
merge1 = lag_feature(merge1, [1], 'date_cat_avg_item_cnt')
# merge1.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

agg = merge1.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
agg.columns = ['date_shop_cat_avg_item_cnt']
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
merge1 = lag_feature(merge1, [1], 'date_shop_cat_avg_item_cnt')
# merge1.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

agg =df_sales_train0.groupby(['item_id']).agg({'item_price': ['mean']})
agg.columns = ['item_avg_item_price']
agg.reset_index(inplace=True)
merge1 = pd.merge(merge1, agg, on=['item_id'], how='left')
agg = df_sales_train0.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
agg.columns = ['date_item_avg_item_price']
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num','item_id'], how='left')
lags = [1,2,3,4,5,6]
merge1 = lag_feature(merge1, lags, 'date_item_avg_item_price')
for i in lags:
    merge1['delta_price_lag_'+str(i)] = \
        (merge1['date_item_avg_item_price_lag_'+str(i)] - merge1['item_avg_item_price']) / merge1['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
merge1['delta_price_lag'] = merge1.apply(select_trend, axis=1)
merge1['delta_price_lag'].fillna(0, inplace=True)
fetures_to_drop = []
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

merge1.drop(fetures_to_drop, axis=1, inplace=True)

agg = df_sales_train0.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
agg.columns = ['date_shop_revenue']
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['date_block_num','shop_id'], how='left')
agg = merge1.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
agg.columns = ['shop_avg_revenue']
agg.reset_index(inplace=True)

merge1 = pd.merge(merge1, agg, on=['shop_id'], how='left')
merge1['delta_revenue'] = (merge1['date_shop_revenue'] - merge1['shop_avg_revenue']) / merge1['shop_avg_revenue']

merge1 = lag_feature(merge1, [1], 'delta_revenue')

merge1['month'] = merge1['date_block_num'] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
merge1['days'] = merge1['month'].map(days)
cache = {}
merge1['item_shop_last_sale'] = -1
for idx, row in merge1.iterrows():    
    key = str(row['item_id'])+' '+str(row['shop_id'])
    if key not in cache:
        if row['item_cnt_month']!=0:
            cache[key] = row['date_block_num']
    else:
        last_date_block_num = cache[key]
        merge1.at[idx, 'item_shop_last_sale'] = row['date_block_num'] - last_date_block_num
        cache[key] = row['date_block_num']  
cache = {}
merge1['item_last_sale'] = -1
for idx, row in merge1.iterrows():    
    key = row['item_id']
    if key not in cache:
        if row['item_cnt_month']!=0:
            cache[key] = row['date_block_num']
    else:
        last_date_block_num = cache[key]
        if row['date_block_num']>last_date_block_num:
            merge1.at[idx, 'item_last_sale'] = row['date_block_num'] - last_date_block_num
            cache[key] = row['date_block_num']   
merge1['item_shop_first_sale'] = merge1['date_block_num'] - merge1.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
merge1['item_first_sale'] = merge1['date_block_num'] - merge1.groupby('item_id')['date_block_num'].transform('min')
merge1['gdp'] = 0
merge1['gdp'][merge1['date_block_num'] < 25 ] = 14101
merge1['gdp'][merge1['date_block_num'] > 24 ] = 9314
merge1.isnull().sum().sum()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
            elif('CPI_inflation' in col):
                df[col].fillna(0, inplace=True)        
    return df
merge1 = fill_na(merge1)
merge1['shop_name'].unique()
merge1['shop_name'].replace('! Yakutsk Ordzhonikidze, 56 Franc' , 'Yakutsk Ordzhonikidze, 56' , inplace=True) 
merge1['shop_name'].replace('! Yakutsk TC "Central" Franc' , 'Yakutsk TC "Central"' , inplace=True) 
merge1['shop_name'].replace('St. Petersburg TK "Nevsky Center"' , 'Petersburg TK "Nevsky Center"' , inplace=True) 
merge1['shop_name'].replace('Shop Online Emergencies' , 'online Shop Emergencies' , inplace=True) 
merge1['shop_name'].replace('Digital storage 1C-line' , 'online Digital storage 1C-line' , inplace=True)
merge1['shop_name'].replace('Zhukovsky Street. Chkalov 39m?' , 'Zhukovsky Street. Chkalov 39m²' , inplace=True)
merge1['city'] = merge1['shop_name'].str.split(' ').map(lambda x: x[0])
merge1['item_category_name'].unique()
merge1['split'] = merge1['item_category_name'].str.split('-')
merge1['category_type'] = merge1['split'].map(lambda x: x[0].strip())
merge1['category_subtype'] = merge1['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
merge1.drop(columns=['item_category_name' , 'split'] , inplace=True , axis=1)
merge1['category_subtype'].replace('Blu','BluRay' , inplace=True)
merge1['category_type'].replace('Movies','Movie' , inplace=True)
merge1['category_type'].unique()
merge1['category_subtype'].unique()
# merge1.to_csv('merge1_Nodummies.csv')
# merge1 = pd.read_csv('merge1_Nodummies.csv' , index_col='Unnamed: 0')
# merge1_Nodummies = pd.read_csv('merge1_Nodummies.csv' , index_col='Unnamed: 0')
merge1_Nodummies - merge1.copy()
merge1_Nodummies = merge1_Nodummies[merge1_Nodummies['date_block_num'] < 34]
merge1_Nodummies = merge1_Nodummies[merge1_Nodummies['item_cnt_month'] > 0]
import plotly.express as px
pie_fig = merge1_Nodummies.groupby('category_type').agg({'item_cnt_month': sum}).\
sort_values(by='item_cnt_month',ascending=False).reset_index()
fig = px.pie(pie_fig, values='item_cnt_month', names='category_type', title='Percentage of Sales Per Category')
fig.show()
import plotly.express as px
sub_pie = merge1_Nodummies.groupby('category_subtype').agg({'item_cnt_month': sum}).\
sort_values(by='item_cnt_month',ascending=False).reset_index()
fig = px.pie(sub_pie, values='item_cnt_month', names='category_subtype', title='Percentage of Sales Per Sub Category')
fig.show()
Z = merge1_Nodummies.groupby('category_type').agg({'item_cnt_month': sum}).sort_values(by='item_cnt_month',ascending=False).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,16))
sns.barplot(data=Z, x='item_cnt_month', y='category_type', palette="magma",orient='h')
plt.yticks()
plt.ylabel('')
plt.xlabel('')
plt.title('Total Categories Sales', fontsize=25)
plt.xticks([i for i in range(0, 350000, 20000)], fontsize=14);
Z = merge1_Nodummies.groupby('category_subtype').agg({'item_cnt_month': sum}).sort_values(by='item_cnt_month',ascending=False).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,16))
sns.barplot(data=Z, x='item_cnt_month', y='category_subtype', palette="magma",orient='h')
plt.yticks(fontsize=16)
plt.ylabel('')
plt.xlabel('')
plt.title('Total Sub Categories Sales', fontsize=25)
plt.xticks([i for i in range(0, 240000, 20000)], fontsize=16);
Z = merge1_Nodummies.groupby('city').agg({'item_cnt_month': sum}).sort_values('item_cnt_month', ascending=False).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,14))
sns.barplot(data=Z, x='item_cnt_month', y='city', palette="gist_earth",orient='h')
plt.ylabel('')
plt.xlabel('')
plt.title('Number of Sales Per City', fontsize=25)
plt.yticks(fontsize=16)
plt.xticks([i for i in range(0, 600000, 30000)]);
p = merge1_Nodummies.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index() 
g = sns.relplot(x="date_block_num", y="item_cnt_month",palette=["b", "r"], ci=None, kind="line", data=p)
g.fig.set_size_inches(15,8)
plt.title('Monthly items sales for 3 years')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.xticks([i for i in range(0, 35)]);
Z = merge1_Nodummies.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
sns.barplot(data=Z, x='date_block_num', y='item_cnt_month', ax = ax, palette="BrBG")
plt.title('Total Sales Per Month', fontsize=25)
plt.xlabel('Months', fontsize=25)
plt.ylabel('Sales', fontsize=25);
Z = merge1_Nodummies.groupby('date_block_num').agg({'item_avg_item_price': sum}).reset_index()
fig, ax = plt.subplots(figsize = (20,10))
sns.barplot(data=Z, x='date_block_num', y='item_avg_item_price', ax = ax, palette="BrBG")
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
plt.title('Total Prices Per Month', fontsize=25)
plt.xlabel('Months', fontsize=25)
plt.ylabel('Price', fontsize=25);
Z = merge1.groupby('date_block_num').agg({'item_id': 'nunique'}).reset_index()
fig, ax = plt.subplots(figsize = (20,10))
sns.barplot(data=Z, x='date_block_num', y='item_id', ax = ax, palette="cool_r")
plt.title('Number of Unique Items', fontsize=25)
plt.xlabel('Months', fontsize=25)
plt.ylabel('Count', fontsize=25);
Z = merge1_Nodummies.groupby('shop_name').agg({'item_cnt_month': sum})\
    .sort_values(by='item_cnt_month', ascending=False).reset_index()
fig, ax = plt.subplots(figsize = (22,18))
sns.barplot(data=Z, x='item_cnt_month', y='shop_name', palette="copper",orient='h', ax=ax)
plt.title('Shops Sales', fontsize=25)
plt.xlabel('')
plt.ylabel('')
plt.yticks(fontsize=18)
plt.xticks([i for i in range(0, 170000, 10000)], fontsize=16);
merge1 = pd.get_dummies(merge1, columns=['city','category_type','category_subtype'],drop_first=True)
# del df_catog
# del df_items
# del df_shops
# del df_sales_train0
# del merge2
# del merge1_Nodummies
merge1.isnull().sum().sort_values().tail(10)
merge1.drop([
             'date_shop_cat_avg_item_cnt' ,
             'date_avg_item_cnt' ,
             'date_item_avg_item_cnt',
             'date_shop_avg_item_cnt',
             'date_cat_avg_item_cnt','shop_name',
             'ID',
             'item_avg_item_price',
             'date_item_avg_item_price',
             'date_shop_revenue',
             'shop_avg_revenue',
             'delta_revenue',
             'delta_revenue_lag_1'
             ], axis=1, inplace=True)
merge1 = merge1[merge1['date_block_num'] > 11]
# Top Correlations

percent=0.20 
cor_train=merge1.corr()
high_corre = cor_train.index[abs(cor_train["item_cnt_month"])>percent]

#to sort columns from highest correlation with item_cnt_month
sorted_cols = cor_train.nlargest(len(high_corre),
'item_cnt_month')['item_cnt_month'].index 

plt.figure(figsize=(15,13))
sns.set(font_scale=1.5)

#plot heatmap with only the top features
nr_corr_matrix = sns.heatmap(merge1[sorted_cols].corr(),
annot=True,cmap="BrBG",square=True, annot_kws={'size':14})
for col in merge1.columns:
    if col == 'date_shop_revenue':
        merge1[col] = merge1[col].astype('float64')
    elif col == 'item_cnt_month':
        merge1[col] = merge1[col].astype('float32')
    elif merge1[col].dtype == 'float64':
        merge1[col] = merge1[col].astype('float16')
    elif col == 'item_id':
        merge1[col] = merge1[col].astype('int16')
    elif merge1[col].dtype == 'int64':
        merge1[col] = merge1[col].astype('int8')
merge1.isnull().sum().sort_values().tail(10)
merge1['item_cnt_month'] = merge1['item_cnt_month'].clip(0,20)
X_plot= merge1[merge1['date_block_num'] < 34]
X = merge1[merge1['date_block_num'] < 34].drop(['item_cnt_month', 'item_name'], axis=1)
y = merge1[merge1['date_block_num']< 34]['item_cnt_month']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.1, random_state = 42)
testing = merge1[merge1['date_block_num'] == 34].drop(['item_cnt_month', 'item_name'], axis=1)
del X
del y
del Z
del cor_train
del pie_fig
del sub_pie
del p
# del cache
# del agg
# del df_sales_train
import gc
gc.collect()
# test size = .1
# shuffle = True
# random_state =42 
# xgboost version = '1.1.0'
# google colab

##### Model
# ts = time.time()

# xgb_model = xgb.XGBRegressor(eta=0.01,
#                                  max_depth=11,n_estimators=1400,
#                                  alpha=2,
#                                  n_jobs=-1,
#                                  tree_method='gpu_hist'
#                                  )

# xgb_hist = xgb_model.fit(X_train,y_train,
#                          eval_set=[(X_train,y_train),(X_test,y_test)],
#                          eval_metric='rmse',
#                          early_stopping_rounds=10)

# time.time() - ts

# results:
# [0]	validation_0-rmse:1.54692	validation_1-rmse:1.52665
# [1399]	validation_0-rmse:0.73776	validation_1-rmse:0.88586
# Train Score: 0.7750457101368959
# Test Score : 0.6667570149949471
# Kaggle Score 0.89995
ts = time.time()

xgb_model = xgb.XGBRegressor(eta=0.01,
                                 max_depth=11,n_estimators=1400,
                                 alpha=2,
                                 n_jobs=-1,
                                 tree_method='gpu_hist'
                                 )

xgb_hist = xgb_model.fit(X_train,y_train,
                         eval_set=[(X_train,y_train),(X_test,y_test)],
                         eval_metric='rmse',
                         early_stopping_rounds=10)


# tree_method='gpu_hist', gpu_id=0
time.time() - ts
y_predV = xgb_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_predV)))
y_pred = xgb_model.predict(testing)
xgb_model.get_params
print('Train Score:', xgb_model.score(X_train, y_train))
print('Test Score :', xgb_model.score(X_test, y_test))
# cv=KFold(n_splits=5, shuffle=True, random_state=1)
# cross_val_score(xgb_model, X, y, cv=cv).mean()
features_importance = xgb_model.get_booster().get_fscore()
f_results = pd.DataFrame(features_importance.items(), columns=['feature_name', 'fscore'])
f_results.sort_values(by='fscore',ascending=False,inplace=True)
f_results.reset_index(inplace=True)
top_features = f_results['feature_name']
top_features
results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('rmse')
plt.title('XGBoost')
plt.show()
fig, ax = plt.subplots(figsize=(14,33))
xgb.plot_importance(xgb_model, ax);
#here we will get item_name from date_block_number 33 (month 33) 
y_predV_Df_test = pd.DataFrame({'item_id' : X_test['item_id'],'shop_id' : X_test['shop_id'] ,'item_cnt_month' : y_predV  }) 
top_10_items = y_predV_Df_test.sort_values(by='item_cnt_month' , ascending=False).head(20)
top_10_items['date_block_num'] = 33 
item_name = []
for i in top_10_items['item_id'].values:
    
    top = pd.merge(top_10_items,merge1[['item_name' , 'item_id','shop_id','date_block_num']],on=['item_id','shop_id','date_block_num'],how='left')#
    
top['date_block_num'] = 34
top.head(10)
y_predV_Df = pd.DataFrame({'date_block_num' : X_test['date_block_num'] ,'item_cnt_month' : y_predV  })
y_test_Df = pd.DataFrame({'date_block_num' : X_test['date_block_num'] ,'item_cnt_month' : y_test  })
fig , ax = plt.subplots(ncols=1 , figsize=(16,8))
xl = y_test_Df.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
y_predsum = y_predV_Df.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
xl.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax ,  linewidth=2 , c='b')
y_predsum.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax , linewidth=2 , c='orange' )
plt.legend(['Actual' , 'Predicted'])
plt.xlabel('Months' , fontsize=25)
# y_predsum.plot(x='date_block_num',y='item_cnt_month', kind='line' , ax=ax)
ax.set_xticks([i for i in range(12, 35)]);
# ax.set_yticks([i for i in range(0, 20000, 3000)]);
y_predDf = pd.DataFrame({'date_block_num' : 34 ,'item_cnt_month' : y_pred  })
y_predsum = y_predDf.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
xl = X_plot.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
mk = pd.concat([xl, y_predsum])
fig , ax = plt.subplots(ncols=1 , figsize=(16,8))
xl = X_plot.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
mk.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax ,  linewidth=2 , c='orange')
xl.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax , linewidth=2 , c='b' )
ax.legend(['Test' , 'Train'])
plt.xlabel('Months' , fontsize=20)
ax.set_xticks([i for i in range(12, 35)]);
fig, ax = plt.subplots(figsize = (20,10))
sns.barplot(data=mk, x='date_block_num', y='item_cnt_month', ax = ax, palette="BrBG")
plt.title('Total Sales Per Month', fontsize=25)
plt.xlabel('Months', fontsize=25)
plt.ylabel('Sales', fontsize=25);
import lightgbm as lgb

ts = time.time()
train_data = lgb.Dataset(data=X_train, label=y_train)
valid_data = lgb.Dataset(data=X_test, label=y_test)

time.time() - ts
    
params = {'num_leaves': 2000, 'max_depth': 19, 'max_bin': 107, 'n_estimators': 1100,
          'bagging_freq': 1, 'bagging_fraction': 0.7135681370918421, 
          'feature_fraction': 0.49446461478601994, 'min_data_in_leaf': 88, 
          'learning_rate': 0.01, 'num_threads': 3, 
          'min_sum_hessian_in_leaf': 6,
         
          'verbosity' : 1,
          'boost_from_average' : 'true',
          'boost' : 'gbdt',
          'metric' : 'rmse',}
lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1, num_boost_round=20)

############################################
# Kaggle Score : 0.90560
# [1]	training's rmse: 1.54864	valid_1's rmse: 1.52789
# [1100]	training's rmse: 0.809505	valid_1's rmse: 0.887232
y_predV = lgb_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_predV)))
y_pred = lgb_model.predict(testing)
thesubmission = df.copy()
thesubmission['item_cnt_month'] = y_pred.clip(0,20)
thesubmission.to_csv('Xgboostlastone.csv', index=False)
thesubmission['item_cnt_month'].head()