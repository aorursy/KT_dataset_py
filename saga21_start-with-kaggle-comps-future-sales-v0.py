import numpy as np 

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import lightgbm as lgb

import datetime

import re

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier as xgb

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor



import os

print(os.listdir("../input"))
##########################################################################################################

##########################################   STEP 1: LOAD DATA   #########################################

##########################################################################################################





sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv", parse_dates=['date'], infer_datetime_format=False, dayfirst=True)

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")



print("Sales_train")

display(sales_train.head(10))

print("Test")

display(test.head(10))

print("Item_categories")

display(item_categories.head(10))

print("Items")

display(items.head(10))

print("Shops")

display(shops.head(1))



# Auxiliar function to reduce data storage

def downcast_dtypes(df):

    # Columns to downcast

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    return df



all_data = sales_train

all_data = downcast_dtypes(all_data)

display(all_data.head(10))



print("Train set size: ", len(sales_train))

print("Test set size: ", len(test))

print("Item categories set size: ", len(item_categories))

print("Items set size: ", len(items))

print("Shops set size: ", len(shops))

print("All data size: ", len(all_data))
##########################################################################################################

######################################   STEP 2: DATA EXPLORATION   ######################################

##########################################################################################################





# Describe merged data to look for inusual values

display(all_data.describe())

#print("Item_price outlier: ")

#print(all_data.loc[all_data['item_price'].idxmax()])

#print("\nItem_cnt_day maximum: ")

#print(all_data.loc[all_data['item_cnt_day'].idxmax()])



f1, axes = plt.subplots(1, 2, figsize=(15,5))

f1.subplots_adjust(hspace=0.4, wspace=0.2)

sns.boxplot(x=all_data['item_price'], ax=axes[0])

sns.boxplot(x=all_data['item_cnt_day'], ax=axes[1])

#sns.boxplot(x=all_data['item_price'], y=all_data['item_cnt_day'], ax=axes[2])



#print(shops['shop_name'].unique())



# Conclusions: 

# 1 - There are negative prices and counts (errors, returns?)

# 2 - Item_id = 6066 has an abnormal large price (item_price = 307980), and is only sold one time

# 3 - 2 items have very large item_cnt_day when compared with the other products

# 4 - Shop_name contains the shops' city names (Москва, Moscow). An additional feature can be obtained

# 5 - Якутск city is expressed as Якутск and !Якутск. This could be fixed

# 6 - Shop_id = 0 & 1 are the same than 57 & 58 but for фран (Google translator => fran maybe franchise). Shop_id = 10 & 11 are the same



# Drop outliers

all_data = all_data.drop(all_data[all_data['item_price']>250000].index)

all_data = all_data.drop(all_data[all_data['item_cnt_day']>800].index)



# Unify duplicated shops

#all_data.loc[all_data['shop_id'] == 57,'shop_id'] = 0

#all_data.loc[all_data['shop_id'] == 58,'shop_id'] = 1

#all_data.loc[all_data['shop_id'] == 11,'shop_id'] = 10



# An alternative for negative price outliers is to replace them with the median value for the impacted shops:

all_data.loc[all_data['item_price'] < 0, 'item_price'] = all_data[(sales_train['shop_id'] == 32) & 

                                                                  (all_data['item_id'] == 2973) & 

                                                                  (all_data['date_block_num'] == 4) & 

                                                                  (all_data['item_price'] > 0)].item_price.median()

print("Raw data length: ",len(sales_train), ", post-outliers length: ", len(all_data))
##########################################################################################################

######################################   STEP 3: MISSINGS CLEANING   #####################################

##########################################################################################################





# Missings count. Surprisingly there are no missings!

missings_count = {col:all_data[col].isnull().sum() for col in all_data.columns}

missings = pd.DataFrame.from_dict(missings_count, orient='index')

print(missings.nlargest(30, 0))
##########################################################################################################

#####################################   STEP 4: FEATURE ENGINEERING   ####################################

##########################################################################################################



def enrich_monthly_data(all_data, sales_train, items, item_categories, shops):

    

    # Extract year-month-day feats

    all_data['year'] = all_data['date'].dt.year

    all_data['month'] = all_data['date'].dt.month

    all_data['day'] = all_data['date'].dt.day



    # Split again data into train (date_block_num: 0-33) and test (date_block_num = 34) 

    all_data_train = all_data[all_data['date_block_num']<34]

    all_data_test = all_data[all_data['date_block_num']==34]



    # Aggregate monthly data and join with items file

    monthly_data = all_data_train.groupby(['month', 'year', 'item_id','shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()

    monthly_data = monthly_data.join(items, on='item_id', rsuffix='_item')

    monthly_data.drop(['item_id_item'], axis=1, inplace=True)



    ## Add median item_price per item, shop and month

    median_item_price = sales_train.groupby(['date_block_num', 'shop_id', 'item_id'],as_index=False).agg({'item_price':{'median_item_price':'median'}})

    monthly_data = pd.merge(monthly_data, median_item_price, on=['item_id', 'shop_id', 'date_block_num'])

    monthly_data = monthly_data.drop(['item_name'], axis=1)

    monthly_data = monthly_data.rename(columns={monthly_data.columns[7]: "item_price"})



    return monthly_data



monthly_data = enrich_monthly_data(all_data, sales_train, items, item_categories, shops)





def obtain_month_columns(monthly_data, test, shops, item_categories):

    

    # Create one column per year/month, with the item_cnt_day per item_id,shop_id

    data_by_month_train = monthly_data.copy()

    data_by_month_train['year/month'] = (monthly_data['year'].map(str)).str.cat(monthly_data['month'].map(str), sep='/')

    data_by_month_train = data_by_month_train[['year/month','item_id','shop_id','item_cnt_day']]

    data_by_month_train = data_by_month_train.pivot_table(index=['item_id','shop_id'], columns='year/month',values='item_cnt_day',fill_value=0).reset_index()



    # Join test data

    data_by_month = pd.merge(test, data_by_month_train, on=['item_id','shop_id'], how='left')

    data_by_month = data_by_month.fillna(0)

    

    # Add item_categories

    data_by_month = data_by_month.join(monthly_data['item_category_id'], on='item_id').join(item_categories, on='item_category_id', rsuffix='_item_categories')

     

    # Extract cities information from shop_name. Replace !Якутск by Якутск since it's the same city

    data_by_month = data_by_month.join(shops, on='shop_id', rsuffix='_shops')

    data_by_month['city'] = data_by_month['shop_name'].str.split(' ').map(lambda row: row[0])

    data_by_month.loc[data_by_month.city == '!Якутск', 'city'] = 'Якутск'

    data_by_month.drop(['shop_id_shops', 'shop_name'], axis=1, inplace=True)

    

    # Extract main category and subcategory from category name

    categories_split = data_by_month['item_category_name'].str.split('-')

    data_by_month['main_category'] = categories_split.map(lambda row: row[0].strip())

    data_by_month['secondary_category'] = categories_split.map(lambda row: row[1].strip())



    # Encode cities and categories

    encoder = sklearn.preprocessing.LabelEncoder()

    data_by_month['city_label'] = encoder.fit_transform(data_by_month['city'])

    data_by_month['main_category_id'] = encoder.fit_transform(data_by_month['main_category'])

    data_by_month['secondary_category_id'] = encoder.fit_transform(data_by_month['secondary_category'])

    data_by_month.drop(['city', 'item_category_name', 'item_category_id_item_categories', 'main_category', 'secondary_category'], axis = 1, inplace = True)

    

    return data_by_month



data_by_month = obtain_month_columns(monthly_data, test, shops, item_categories)

data_by_month = downcast_dtypes(data_by_month)
data_by_month
###############################################################################################################

###################################       STEP 5: DATASET PROCESSING        ###################################

###############################################################################################################



# Rename data

X_test_full = data_by_month.copy()



print(len(X_test_full))



# Break off test set from training data

month_to_predict = '2015/10'

y_train = data_by_month[month_to_predict]

X_train_full = data_by_month.drop(labels=[month_to_predict], axis=1)



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 40 and 

                        X_train_full[cname].dtype not in ['int64', 'float64', 'int32', 'float32']]

print("Low cardinality columns: ", low_cardinality_cols)



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64', 'int32', 'float32']]

print("Numeric columns: ", numeric_cols)



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data if needed

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
# Define model with best MAE

#model = xgb(colsample_bytree=0.7, learning_rate=.01, max_depth=5, min_child_weight=3, n_estimators=100, 

#                     nthread=1, objective='reg:squarederror', subsample=0.7, random_state=21, 

#                     early_stopping_rounds = 10, eval_set=[(X_valid, y_valid)], verbose=False)
X_train.columns
model=LGBMRegressor(

        n_estimators=200,

        learning_rate=0.03,

        num_leaves=32,

        colsample_bytree=0.9497036,

        subsample=0.8715623,

        max_depth=8,

        reg_alpha=0.04,

        reg_lambda=0.073,

        min_split_gain=0.0222415,

        min_child_weight=40)
model.fit(X_train, y_train,)
# Cross validation accuracy for 3 folds

scores = cross_val_score(model, X_train, y_train, cv=3)

print(scores)
# Get the test set predictions and clip values to the specified range

y_pred = model.predict(X_test).clip(0., 20.)



# make sure results are in the same order as the original test set

(test[['shop_id','item_id']].values == X_test[['shop_id','item_id']]).all()



# Create the submission file and submit!

preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])

preds.to_csv('submission.csv',index_label='ID')
print(len(test[['shop_id','item_id']].values))

print(len(X_test[['shop_id','item_id']]))

print(len(X_test_full))
X_test