import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime as dt
import gc
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import  mean_squared_error
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt
% matplotlib inline
train_df = pd.read_csv("../input/sales-train/sales_train_v2.csv")
test_df = pd.read_csv("../input/exploresales/test.csv")
item_category_df = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items_df = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
print(train_df.shape)
train_df.head()
print(test_df.shape)
test_df.head()
print(items_df.shape)
items_df.head()
print(item_category_df.shape)
item_category_df.head()
# Get english category names
cat_list = list(item_category_df.item_category_name)

cat_list[0] = 'PC Headsets'
for i in range(1,8):
    cat_list[i] = 'Access'
cat_list[8] = "Ticket"
cat_list[9] = "Deliveries"
for i in range(10,18):
    cat_list[i] = 'Consoles'
for i in range(18,25):
    cat_list[i] = 'Consoles Games'
cat_list[25] = "Game Accessories"
for i in range(26,28):
    cat_list[i] = 'phone games'
for i in range(28,32):
    cat_list[i] = 'CD games'
for i in range(32,37):
    cat_list[i] = 'Card'
for i in range(37,43):
    cat_list[i] = 'Movie'
for i in range(43,55):
    cat_list[i] = 'Books'
for i in range(55,61):
    cat_list[i] = 'Music'
for i in range(61,73):
    cat_list[i] = 'Gifts'
for i in range(73,79):
    cat_list[i] = 'Soft'
for i in range(79,81):
    cat_list[i] = 'System Tools'
for i in range(81,83):
    cat_list[i] = 'Clean media'
cat_list[83] = "Elements of a food"

item_category_df['category'] = cat_list
item_cat_df = pd.merge(item_category_df, items_df, how="inner", on="item_category_id")[["item_id", "category"]]
item_cat_df.head()
# convert "date" from str to date
# train_df["date"] = pd.to_datetime(train_df["date"])
# make pivot table based on "shop_id" and "item_id"
train_pivot_df = train_df.pivot_table(index=["shop_id", "item_id"], 
                                      columns="date_block_num", 
                                      values="item_cnt_day", 
                                      aggfunc="sum").fillna(0).reset_index()
train_pivot_df = pd.merge(train_pivot_df, item_cat_df, how="inner", on="item_id")
# encode label to "category"
le = preprocessing.LabelEncoder()
train_pivot_df["category"] = le.fit_transform(train_pivot_df["category"])
train_pivot_df = train_pivot_df[["shop_id", "item_id", "category"] + list(range(34))]
train_pivot_df["shop_id"] = train_pivot_df["shop_id"].astype("str")
train_pivot_df["item_id"] = train_pivot_df["item_id"].astype("str")
train_pivot_df["category"] = train_pivot_df["category"].astype("str")
train_pivot_df.head()
Train_df, Validate_df = train_test_split(train_pivot_df, test_size = 0.3, random_state = 1234)
X_train = Train_df.iloc[:, :-1].values
X_validate = Validate_df.iloc[:, :-1].values

y_train = Train_df.iloc[:, -1].values
y_validate = Validate_df.iloc[:, -1].values
print(X_train.shape, X_validate.shape, y_train.shape, y_validate.shape)
# Cross Validation
params = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 50]}
rf = RandomForestRegressor(random_state=1234)
gscv = GridSearchCV(rf, param_grid=params, verbose=1, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

start = time.time()
gscv.fit(X_train, y_train)
print(time.time() - start)
gscv.best_params_
# Train the data
rf = RandomForestRegressor(n_estimators=gscv.best_params_["n_estimators"], 
                           max_depth=gscv.best_params_["max_depth"], random_state=1234)
rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
rf_mse_train = mean_squared_error(y_pred_train, y_train)
print("RF Train-data RMSE", np.sqrt(rf_mse_train))
# Validate the data
y_pred_validate = rf.predict(X_validate)
rf_mse_validate = mean_squared_error(y_pred_validate, y_validate)
print("RF Valdation-data RMSE", np.sqrt(rf_mse_validate))
# processing test data
test_df["shop_id"] = test_df["shop_id"].astype("str")
test_df["item_id"] = test_df["item_id"].astype("str")
test_pivot_df = pd.merge(test_df, train_pivot_df, how="left", on=["shop_id", "item_id"]).fillna(0)
test_pivot_df.columns = list(test_pivot_df.columns[:4]) + list(range(-1,33))
test_pivot_df.head()
X_test = test_pivot_df.drop(["ID", -1], axis=1).values
print(X_test.shape)
X_test[0:1]
y_pred = rf.predict(X_test)
submission_df = pd.DataFrame({'ID':test_df["ID"], 'item_cnt_month': y_pred.clip(0. ,20.)})
submission_df.to_csv('sub_rf_v1.csv',index=False)  # RMSE: 1.18969
