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
import matplotlib.pyplot as plt
%matplotlib inline
from itertools import product
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import plotly.express as px
import time
# !pip install googletrans
# from googletrans import Translator
# translator= Translator()
# translations = {}
# # unique elements of the column
# unique_elements = df_items['item_name'].unique()
# for element in unique_elements:
#     # add translation to the dictionary
#     translations[element] = translator.translate(element).text
# df_items['item_name'].replace(result_translation, inplace = True)
df_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
df_shops = pd.read_csv('/kaggle/input/predict/shops_en.csv')
df_items = pd.read_csv('/kaggle/input/predict/items_en.csv')
df_catog = pd.read_csv('/kaggle/input/predict/item_categories_en.csv')
df_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
# Find specific column lag with respect to date
def feature_lag(dataframe, lags, column):
    temp = dataframe[['date_block_num','shop_id','item_id',column]]
    for i in lags:
        shifted = temp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', column+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        dataframe = pd.merge(dataframe, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return dataframe
# Show first 5 columns of "df_train" Dataframe
df_train.head()
# Show description of "df_train" Dataframe
df_train.describe().T
# Show unique values of each column of "df_train" Dataframe
df_train.nunique()
# Show first 5 columns of "df_test" Dataframe
df_test.head()
# Show unique values of each column of "df_test" Dataframe
df_test.nunique()
# Show first 5 columns of "df_shops" Dataframe
df_shops.head()
# Show unique values of each column of "df_shops" Dataframe
df_shops.nunique()
# Show first 5 columns of "df_items" Dataframe
df_items.head()
# Show unique values of each column of "df_items" Dataframe
df_items.nunique()
# Show first 5 columns of "df_catog" Dataframe
df_catog.head()
# Show unique values of each column of "df_catog" Dataframe
df_catog.nunique()
# Show first 60 columns of "df_shops" Dataframe
df_shops.head(60)
# Removing Duplicates
# 1) ! Yakutsk Ordzhonikidze, 56 Franc ---> Yakutsk Ordzhonikidze, 56
df_train.loc[df_train.shop_id == 0, "shop_id"] = 57
df_test.loc[df_test.shop_id == 0 , "shop_id"] = 57
# 2) ! Yakutsk TC "Central" Franc  ---> Yakutsk TC "Central"
df_train.loc[df_train.shop_id == 1, "shop_id"] = 58
df_test.loc[df_test.shop_id == 1 , "shop_id"] = 58
# 3) Zhukovsky Street. Chkalov 39m²  ---> Zhukovsky Street. Chkalov 39m?
df_train.loc[df_train.shop_id == 11, "shop_id"] = 10
df_test.loc[df_test.shop_id == 11, "shop_id"] = 10
# 4) RostovNaDonu TRC "Megacenter Horizon" Island ---> RostovNaDonu TRC "Megacenter Horizon"
df_train.loc[df_train.shop_id == 40, "shop_id"] = 39
df_test.loc[df_test.shop_id == 40, "shop_id"] = 39
# Box plot of item_price Column in "df_train" Dataframe to check the outliers 
df_train['item_price'].plot(kind="box")
# Box plot of item_cnt_day Column in "df_train" Dataframe to check the outliers
df_train['item_cnt_day'].plot(kind="box")
# Removing Outliers from "df_train" Dataframe
df_train = df_train[df_train['item_price']<100000]
df_train = df_train[df_train['item_cnt_day']<1001]
# Checking Negative values item_price column in "df_train" Dataframe
df_train_price_negative = df_train[df_train['item_price']<=0]
df_train_price_negative
# Removing Negative values item_price column in "df_train" Dataframe
df_train = df_train[df_train['item_price'] > 0].reset_index(drop = True)
# Checking Negative values item_cnt_day column in "df_train" Dataframe
df_train_cnt_negative = df_train[df_train['item_cnt_day']<0]
df_train_cnt_negative.count()
# Changing negative values of item_cnt_day in "df_train" with zero
df_train.loc[df_train['item_cnt_day'] < 1, "item_cnt_day"] = 0
# Converting day by day data into per month data 
temp = []
cols = ['date_block_num','shop_id','item_id']
for i in df_train['date_block_num'].unique():
    sales = df_train[df_train['date_block_num'] == i]
    temp.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique()))))
    
updated_df_train = pd.DataFrame(np.vstack(temp), columns=cols)
# length of new dataframe
len(updated_df_train)
# Addind new column revenue in "updated_df_train" Dataframe
df_train['revenue'] = df_train['item_price'] *  df_train['item_cnt_day']
# Show first 5 columns of "df_train" Dataframe
df_train.head()
# Group by 'date_block_num','shop_id','item_id' and count item_cnt_month
groupby_date_shop_item = df_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
groupby_date_shop_item.columns = ['item_cnt_month']
groupby_date_shop_item.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, groupby_date_shop_item, on=cols, how='left').fillna(0)
# Show first 5 columns of "updated_df_train" Dataframe
updated_df_train.head()
# Training Dataframe have 33 months so next is 34Th month and test dataframe have only one month data 
df_test['date_block_num'] = 34
# merging df_test into updated_df_train 
test_shop_ids = df_test['shop_id'].unique()
test_item_ids = df_test['item_id'].unique()
updated_df_train = updated_df_train[updated_df_train['shop_id'].isin(test_shop_ids)]
updated_df_train = updated_df_train[updated_df_train['item_id'].isin(test_item_ids)]
updated_df_train.reset_index(inplace=True, drop=True)
updated_df_train = pd.concat([updated_df_train, df_test], ignore_index=True, sort=False, keys=['date_block_num','shop_id','item_id'])
updated_df_train.fillna(0, inplace=True)
# merging df_shops, df_items, df_catog into updated_df_train 
updated_df_train = pd.merge(updated_df_train, df_shops, on=['shop_id'], how='left')
updated_df_train = pd.merge(updated_df_train, df_items, on=['item_id'], how='left')
updated_df_train = pd.merge(updated_df_train, df_catog, on=['item_category_id'], how='left')
# Show first 5 columns of "updated_df_train" Dataframe
updated_df_train.head()
# show unique item_catogories name in "Updated_df_train" dataframe
updated_df_train['item_category_name'].unique()
# splitting item_category_name column into category_type and category_subtype
updated_df_train['split'] = updated_df_train['item_category_name'].str.split('-')
updated_df_train['category_type'] = updated_df_train['split'].map(lambda x: x[0].strip())
updated_df_train['category_subtype'] = updated_df_train['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
updated_df_train.drop(columns=['item_category_name' , 'split'] , inplace=True , axis=1)
# correcting category_subtype name and category_type column values
updated_df_train['category_subtype'].replace('Blu','BluRay' , inplace=True)
updated_df_train['category_type'].replace('Movies','Movie' , inplace=True)
updated_df_train['category_type'].replace('Игры','Games' , inplace=True)
# Show first 5 columns of "updated_df_train" Dataframe
updated_df_train.head()
# Correcting shop_name column values
updated_df_train['shop_name'].replace('St. Petersburg TK "Nevsky Center"' , 'Petersburg TK "Nevsky Center"' , inplace=True) 
updated_df_train['shop_name'].replace('Shop Online Emergencies' , 'online Shop Emergencies' , inplace=True) 
updated_df_train['shop_name'].replace('Digital storage 1C-line' , 'online Digital storage 1C-line' , inplace=True)
# getting city name from shop_name column
updated_df_train['city'] = updated_df_train['shop_name'].str.split(' ').map(lambda x: x[0])
# Adding new column avg_item_cnt_per_mnth column which tells us average of all items sales per month
temp = updated_df_train.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
temp.columns = [ 'avg_item_cnt_per_mnth' ]
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num'], how='left')
# Adding new column "date_item_avg_item_cnt"  which tells us average for each item sales per month
temp = updated_df_train.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
temp.columns = [ 'date_item_avg_item_cnt' ]
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num','item_id'], how='left')
# Adding new column "date_shop_avg_item_cnt" which tells us average for each shop sales per month
temp = updated_df_train.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
temp.columns = [ 'date_shop_avg_item_cnt' ]
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num','shop_id'], how='left')
updated_df_train.columns
# Adding new column "date_cat_avg_item_cnt" ahich tells us average for each category sales per month
temp = updated_df_train.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
temp.columns = [ 'date_cat_avg_item_cnt' ]
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num','item_category_id'], how='left')
# Adding new column "date_shop_cat_avg_item_cnt" which tells us average for each shop with category sales per month
temp = updated_df_train.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
temp.columns = ['date_shop_cat_avg_item_cnt']
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
# Adding new column "item_avg_item_price" which tells us average price for each item
temp =df_train.groupby(['item_id']).agg({'item_price': ['mean']})
temp.columns = ['item_avg_item_price']
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['item_id'], how='left')
# Adding new column "date_item_avg_item_price" which tells us average price for each item per month
temp = df_train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
temp.columns = ['date_item_avg_item_price']
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num','item_id'], how='left')
# Adding new column "date_shop_revenue" which tells us sum of revenue for each shop per month
temp = df_train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
temp.columns = ['date_shop_revenue']
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['date_block_num','shop_id'], how='left')
# Adding  new column "date_shop_revenue" which tells us avreage of revenue for each shop
temp = updated_df_train.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
temp.columns = ['shop_avg_revenue']
temp.reset_index(inplace=True)
updated_df_train = pd.merge(updated_df_train, temp, on=['shop_id'], how='left')
# Adding  new lag columns which will contains the percent of the difference-
# -between average shop revenue in lag months and sum of revenue for each shop per month
updated_df_train['delta_revenue'] = (updated_df_train['date_shop_revenue'] - updated_df_train['shop_avg_revenue']) / updated_df_train['shop_avg_revenue']
# Adding new column "month" wich will take month number from date_block_num
updated_df_train['month'] = updated_df_train['date_block_num'] % 12
plot_fig = updated_df_train.groupby('month').agg({'item_cnt_month': sum}).sort_values(by='item_cnt_month',ascending=False).reset_index()

plt.figure(figsize=(13,8))
fig = sns.barplot(x="month", y="item_cnt_month", data=plot_fig)

plt.xlabel("Item Count per Month", fontsize=12)
plt.ylabel("Months", fontsize=12)
plt.title("Monthly Sales", fontsize=18)
plt.show(fig)
# Adding Lag Columns

updated_df_train = feature_lag(updated_df_train, [1], 'avg_item_cnt_per_mnth')
# updated_df_train.drop(['date_avg_item_cnt'], axis=1, inplace=True)
updated_df_train = feature_lag(updated_df_train, [1,2,3,6,12], 'date_item_avg_item_cnt')
# updated_df_train.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
updated_df_train = feature_lag(updated_df_train, [1,2,3,6,12], 'date_shop_avg_item_cnt')
# updated_df_train.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
updated_df_train = feature_lag(updated_df_train, [1,2,3,6,12], 'date_item_avg_item_cnt')
# updated_df_train.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
updated_df_train = feature_lag(updated_df_train, [1,2,3,6,12], 'date_shop_avg_item_cnt')
# updated_df_train.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
updated_df_train = feature_lag(updated_df_train, [1], 'date_cat_avg_item_cnt')
# updated_df_train.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
updated_df_train = feature_lag(updated_df_train, [1], 'date_shop_cat_avg_item_cnt')
# updated_df_train.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

updated_df_train = feature_lag(updated_df_train, [1,2,3,4,5,6], 'date_item_avg_item_price')
updated_df_train = feature_lag(updated_df_train, [1], 'delta_revenue')
lags = [1,2,3,4,5,6]
for i in lags:
    updated_df_train['delta_price_lag_'+str(i)] = (updated_df_train['date_item_avg_item_price_lag_'+str(i)] - updated_df_train['item_avg_item_price']) / updated_df_train['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
updated_df_train['delta_price_lag'] = updated_df_train.apply(select_trend, axis=1)
updated_df_train['delta_price_lag'].fillna(0, inplace=True)

fetures_to_drop = []
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

updated_df_train.drop(fetures_to_drop, axis=1, inplace=True)

updated_df_train.columns
# Monthly Sales per category type
plot_fig = updated_df_train.groupby('category_type').agg({'item_cnt_month': sum}).sort_values(by='item_cnt_month',ascending=False).reset_index()
plt.figure(figsize=(13,8))
fig = sns.barplot(x="item_cnt_month", y="category_type", data=plot_fig)

plt.xlabel("Item Count per Month", fontsize=12)
plt.ylabel("Category Type", fontsize=12)
plt.title("Monthly Sales per category type", fontsize=18)
plt.show(fig)
# Sales Percentage Per Category
plt.figure(figsize=(20,10))
fig = px.pie(plot_fig, values='item_cnt_month', names='category_type', title='Sales Percentage Per Category')
fig.show()
# Monthly Sales Per Sub Category Type
plot_fig = updated_df_train.groupby('category_subtype').agg({'item_cnt_month': sum}).sort_values(by='item_cnt_month',ascending=False).reset_index()
plt.figure(figsize=(13,9))
fig = sns.barplot(x="item_cnt_month", y="category_subtype", data=plot_fig)

plt.xlabel("Item Count per Month", fontsize=12)
plt.ylabel("Sub Category Type", fontsize=12)
plt.title("Monthly Sales Per Sub Category Type", fontsize=18)
plt.show(fig)
# Sales Percentage Per Sub Category
plt.figure(figsize=(20,10))
fig = px.pie(plot_fig, values='item_cnt_month', names='category_subtype', title='Sales Percentage Per Sub Category')
fig.show()
# Total Sales Per City
plot_fig = updated_df_train.groupby('city').agg({'item_cnt_month': sum}).sort_values('item_cnt_month', ascending=False).reset_index()
plt.figure(figsize=(13,9))
fig = sns.barplot(x="item_cnt_month", y="city", data=plot_fig)
plt.xlabel("Item Count per Month", fontsize=12)
plt.ylabel("City", fontsize=12)
plt.title("Total Sales Per City", fontsize=18)
plt.show(fig)
# Sales Percentage Per City
plt.figure(figsize=(20,10))
fig = px.pie(plot_fig, values='item_cnt_month', names='city', title='Sales Percentage Per City')
fig.show()
# Total Sales Per Month
plot_fig = updated_df_train.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index() 
plot_fig = plot_fig[plot_fig['date_block_num']<34]
plt.figure(figsize=(13,8))
fig = sns.lineplot(x="date_block_num", y="item_cnt_month", data=plot_fig)
plt.xlabel("Item Count per Month", fontsize=12)
plt.ylabel("Months", fontsize=12)
plt.title("Total Sales Per Month", fontsize=18)
plt.xticks([i for i in range(0, 34)])
plt.show(fig)

# Avg Price Per Month
plot_fig = updated_df_train.groupby('date_block_num').agg({'item_avg_item_price': sum}).reset_index()
plot_fig = plot_fig[plot_fig['date_block_num']<34]
plt.figure(figsize=(13,9))
fig = sns.barplot(x="date_block_num", y="item_avg_item_price", data=plot_fig)
plt.xlabel("Months", fontsize=12)
plt.ylabel("Avg Price", fontsize=12)
plt.title("Avg Price Per Month", fontsize=18)
plt.show(fig)
# Shop Sales
plot_fig =  updated_df_train.groupby('shop_name').agg({'item_cnt_month': sum}).sort_values(by='item_cnt_month', ascending=False).reset_index()
plt.figure(figsize=(13,8))
fig = sns.barplot(x="item_cnt_month", y="shop_name", data=plot_fig)
plt.xlabel("Item Count per Month", fontsize=12)
plt.ylabel("Shop Name", fontsize=12)
plt.title("Shop Sales", fontsize=18)
plt.show(fig)
# Description of updated_df_train
updated_df_train.describe().T
# Label encoding Categorical Features
cols=updated_df_train.select_dtypes(include=['object']).columns
le=LabelEncoder()
for i in cols:
    updated_df_train[i]=le.fit_transform(updated_df_train[i])
updated_df_train.columns
# Droping Unnecessary Features
updated_df_train.drop(['date_shop_cat_avg_item_cnt', 'avg_item_cnt_per_mnth' ,'date_item_avg_item_cnt',
                        'date_shop_avg_item_cnt','date_cat_avg_item_cnt','shop_name','ID','item_avg_item_price',
                        'date_item_avg_item_price','date_shop_revenue','shop_avg_revenue','delta_revenue',
                        'delta_revenue_lag_1'], axis=1, inplace=True)
# we will remove first 12 months because we are using 12 as lag
updated_df_train = updated_df_train[updated_df_train['date_block_num'] > 11]
# Our data is too big so to save memory and modeling time we will change types for all columns
for col in updated_df_train.columns:
    if col == 'date_shop_revenue':
        updated_df_train[col] = updated_df_train[col].astype('float64')
    elif col == 'item_cnt_month':
        updated_df_train[col] = updated_df_train[col].astype('float32')
    elif updated_df_train[col].dtype == 'float64':
        updated_df_train[col] = updated_df_train[col].astype('float16')
    elif col == 'item_id':
        updated_df_train[col] = updated_df_train[col].astype('int16')
    elif updated_df_train[col].dtype == 'int64':
        updated_df_train[col] = updated_df_train[col].astype('int8')
# AS mention in problem to clip the item_cnt_month to (0,20)
updated_df_train['item_cnt_month'] = updated_df_train['item_cnt_month'].clip(0,20)
# Dividing the Features into X,y
X = updated_df_train[updated_df_train['date_block_num'] < 34].drop(['item_cnt_month', 'item_name'], axis=1)
y = updated_df_train[updated_df_train['date_block_num']< 34]['item_cnt_month']
# Spliting dataframe into train and test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.1, random_state = 56)
# for testing
test = updated_df_train[updated_df_train['date_block_num'] == 34].drop(['item_cnt_month', 'item_name'], axis=1)
# for garbage collection
import gc
gc.collect()
# applying Model Xgboost
ts = time.time()

xgb_model = xgb.XGBRegressor(eta=0.01,
                                 max_depth=10,n_estimators=2000,
                                 colsample_bytree=0.5,
                                 subsample=0.8,
                                 gamma=2, reg_alpha=0, reg_lambda=2, min_child_weight=300,
                                 max_bin=2048,
                                 n_jobs=-1,
                                 tree_method='hist'
                                 )

xgb_hist = xgb_model.fit(X_train,y_train,
                         eval_set=[(X_train,y_train),(X_test,y_test)],
                         eval_metric='rmse',
                         early_stopping_rounds=10)

time.time() - ts
# prediction RMS value
y_predV = xgb_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_predV)))
# prediction on test
y_pred = xgb_model.predict(test)
# Model Scores 
print('Train Score:', xgb_model.score(X_train, y_train))
print('Test Score :', xgb_model.score(X_test, y_test))
results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# Decay in RMS value over n_estimators
fig, ax = plt.subplots(figsize=(13,8))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('rmse', fontsize=12)
plt.xlabel('n_estimators', fontsize=12)
plt.title('XGBoost', fontsize=18)
plt.show()
y_predV_Df = pd.DataFrame({'date_block_num' : X_test['date_block_num'] ,'item_cnt_month' : y_predV  })
y_test_Df = pd.DataFrame({'date_block_num' : X_test['date_block_num'] ,'item_cnt_month' : y_test  })
# showing actual and predicted values
fig , ax = plt.subplots(ncols=1 , figsize=(13,8))
xl = y_test_Df.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
y_predsum = y_predV_Df.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
xl.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax ,  linewidth=2 , c='b')
y_predsum.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax , linewidth=2 , c='orange' )
plt.legend(['Actual' , 'Predicted'])
plt.xlabel('Months' , fontsize=12)
plt.ylabel('Item Count per Month' , fontsize=12)
plt.title('Total Sales Per Month', fontsize=18)
ax.set_xticks([i for i in range(12, 35)])
plt.show()
X_plot= updated_df_train[updated_df_train['date_block_num'] < 34]
y_predDf = pd.DataFrame({'date_block_num' : 34 ,'item_cnt_month' : y_pred  })
y_predsum = y_predDf.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
xl = X_plot.groupby('date_block_num').agg({'item_cnt_month': sum}).reset_index()
mk = pd.concat([xl, y_predsum])
# ploting prediction sales
fig , ax = plt.subplots(ncols=1 , figsize=(13,8))
mk.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax ,  linewidth=2 , c='orange')
xl.plot(x='date_block_num',y='item_cnt_month' , kind='line' , ax=ax , linewidth=2 , c='b' )
ax.legend(['Test' , 'Train'])
plt.xlabel('Months' , fontsize=12)
plt.ylabel('Item Count per Month' , fontsize=12)
plt.title('Total Sales Per Month', fontsize=18)
ax.set_xticks([i for i in range(12, 35)])
plt.show()
# predicted Total Sales Per Month
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (13,8))
sns.barplot(data=mk, x='date_block_num', y='item_cnt_month', ax = ax)
plt.title('Total Sales Per Month', fontsize=18)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Sales', fontsize=12)
df_test.head()

df_test['item_cnt_month'] = y_pred
df_test.loc[:,['ID','item_cnt_month']].to_csv('Submission_boost.csv',index = False)
df_test.head()
