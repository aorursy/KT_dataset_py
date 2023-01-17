import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import pickle

from statsmodels.api import OLS

import statsmodels.api as sm

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor



items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

item_categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

sample_submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

sales_test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

sales_train
sns.boxplot(sales_train["item_cnt_day"])
sns.boxplot(sales_train["item_price"])
sales_train["item_cnt_day"] = sales_train["item_cnt_day"].clip(0,200)

sales_train["item_price"] = sales_train["item_price"].clip(0,5000)
# item counts over months

sales_train.groupby("date_block_num")["item_cnt_day"].sum().plot()
# distribution of shop_item combinations in train and test set

# only a few shops/items are in test and train set

# no leakage can be identified, split seems random



train_unique = sales_train.groupby(["shop_id", "item_id"]).size()

test_unique = sales_test.groupby(["shop_id", "item_id"]).size()



train_unique = pd.DataFrame({"in_train":True}, index=train_unique.index)

test_unique = pd.DataFrame({"in_test":True}, index=test_unique.index)



combined = pd.merge(train_unique, test_unique, on=["shop_id", "item_id"], how="outer").fillna(False)



combined["in_both"] = combined["in_train"] & combined["in_test"]



num_in_both = sum(combined["in_both"] == True)

num_in_train = sum((combined["in_train"] == True) & (combined["in_test"] == False))

num_in_test = sum((combined["in_test"] == True) & (combined["in_train"] == False))



pd.DataFrame({"in_both":[num_in_both], 

              "in_train": [num_in_train],

              "in_test": [num_in_test]}).T.plot.pie(subplots=True)
%%time

#Process data into nice represenation of counts for each item, shop and date_block combination

from itertools import product



matrix = []

index_cols  = ["date_block_num", "shop_id", "item_id"]

for i in range(34):

    sales = sales_train[sales_train.date_block_num == i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype=np.int16))



matrix = pd.DataFrame(np.vstack(matrix), columns=index_cols)

matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)

matrix["shop_id"] = matrix["shop_id"].astype(np.int8)

matrix["item_id"] = matrix["item_id"].astype(np.int16)

matrix.sort_values(index_cols, inplace = True )

matrix
# add target variable (item counts for each month) to data

item_counts = sales_train.groupby(index_cols).agg({"item_cnt_day": ["sum"]}).reset_index()



matrix = pd.merge(matrix, item_counts, on=index_cols, how="left")

matrix.fillna(0, inplace=True)

matrix.rename(columns={matrix.columns[-1]: "item_cnt_month"}, inplace=True)

matrix["item_cnt_month"] = matrix["item_cnt_month"].clip(0,20).astype(np.float16)

matrix
# add test data to matrix

sales_test["date_block_num"] = 34

matrix = pd.concat([matrix, sales_test.drop(["ID"], axis=1)], ignore_index=True, keys=index_cols)

matrix.fillna(0, inplace=True)

matrix
# downgrade types

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.dtypes
# lag features and mean encoding



def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id', col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df



lag_dates = [1,2,3,6,12]



matrix = lag_feature(matrix, lag_dates, "item_cnt_month")

matrix.fillna(0, inplace=True)



# mean item count per month

feature = matrix.groupby(['date_block_num']).agg({'item_cnt_month': 'mean'})

feature.index = feature.index+1

feature.columns = ["item_cnt_month_mean_lag_1"]

matrix = pd.merge(matrix, feature, on=["date_block_num"], how="left")



# item category

matrix = pd.merge(matrix, items.drop(["item_name"], axis=1), how="left", on="item_id")



# mean count of item_id

feature = matrix.groupby(['date_block_num', "item_id"]).agg({'item_cnt_month': 'mean'}).reset_index()

feature

feature.rename({"item_cnt_month": "means_cnt_item_id_lag_1"}, axis=1, inplace=True)

feature["date_block_num"] += 1

feature

matrix = pd.merge(matrix, feature, on=["date_block_num", "item_id"], how="left")



# mean count of shop_id

feature = matrix.groupby(['date_block_num', "shop_id"]).agg({'item_cnt_month': 'mean'}).reset_index()

feature

feature.rename({"item_cnt_month": "means_cnt_shop_id_lag_1"}, axis=1, inplace=True)

feature["date_block_num"] += 1

feature

matrix = pd.merge(matrix, feature, on=["date_block_num", "shop_id"], how="left")



# number of counts for shop/item id combination per date_block_num

piv = sales_train.pivot_table(values="item_price", index=["shop_id", "item_id"] ,columns="date_block_num", aggfunc="mean")

piv = piv.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)

piv_diff = piv.diff(axis=1)

piv_diff.columns += 1

piv_div = piv_diff.stack()

piv_div.name = "price_diff_lag_1"

matrix = pd.merge(matrix, piv_div, on=["shop_id", "item_id", "date_block_num"], how="left")



matrix.fillna(0, inplace=True)
# textual and coordinate features from shop and category



shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())

shops.loc[shops.city == '!якутск', 'city'] = 'якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])



coords = dict()

coords['якутск'] = (62.028098, 129.732555, 4)

coords['адыгея'] = (44.609764, 40.100516, 3)

coords['балашиха'] = (55.8094500, 37.9580600, 1)

coords['волжский'] = (53.4305800, 50.1190000, 3)

coords['вологда'] = (59.2239000, 39.8839800, 2)

coords['воронеж'] = (51.6720400, 39.1843000, 3)

coords['выездная'] = (0, 0, 0)

coords['жуковский'] = (55.5952800, 38.1202800, 1)

coords['интернет-магазин'] = (0, 0, 0)

coords['казань'] = (55.7887400, 49.1221400, 4)

coords['калуга'] = (54.5293000, 36.2754200, 4)

coords['коломна'] = (55.0794400, 38.7783300, 4)

coords['красноярск'] = (56.0183900, 92.8671700, 4)

coords['курск'] = (51.7373300, 36.1873500, 3)

coords['москва'] = (55.7522200, 37.6155600, 1)

coords['мытищи'] = (55.9116300, 37.7307600, 1)

coords['н.новгород'] = (56.3286700, 44.0020500, 4)

coords['новосибирск'] = (55.0415000, 82.9346000, 4)

coords['омск'] = (54.9924400, 73.3685900, 4)

coords['ростовнадону'] = (47.2313500, 39.7232800, 3)

coords['спб'] = (59.9386300, 30.3141300, 2)

coords['самара'] = (53.2000700, 50.1500000, 4)

coords['сергиев'] = (56.3000000, 38.1333300, 4)

coords['сургут'] = (61.2500000, 73.4166700, 4)

coords['томск'] = (56.4977100, 84.9743700, 4)

coords['тюмень'] = (57.1522200, 65.5272200, 4)

coords['уфа'] = (54.7430600, 55.9677900, 4)

coords['химки'] = (55.8970400, 37.4296900, 1)

coords['цифровой'] = (0, 0, 0)

coords['чехов'] = (55.1477000, 37.4772800, 4)

coords['ярославль'] = (57.6298700, 39.8736800, 2) 



shops['city_coord_1'] = shops['city'].apply(lambda x: coords[x][0])

shops['city_coord_2'] = shops['city'].apply(lambda x: coords[x][1])

shops['country_part'] = shops['city'].apply(lambda x: coords[x][2])



shops = shops[['shop_id', 'city_code', 'city_coord_1', 'city_coord_2', 'country_part']]



matrix = pd.merge(matrix, shops, on=["shop_id"], how="left")
# features about items and their category names (common category and category code)



map_dict = {

            'Чистые носители (штучные)': 'Чистые носители',

            'Чистые носители (шпиль)' : 'Чистые носители',

            'PC ': 'Аксессуары',

            'Служебные': 'Служебные '

            }



items = pd.merge(items, item_categories, on='item_category_id')



items['item_category'] = items['item_category_name'].apply(lambda x: x.split('-')[0])

items['item_category'] = items['item_category'].apply(lambda x: map_dict[x] if x in map_dict.keys() else x)

items['item_category_common'] = LabelEncoder().fit_transform(items['item_category'])



items['item_category_code'] = LabelEncoder().fit_transform(items['item_category_name'])

items = items[['item_id', 'item_category_common', 'item_category_code']]



matrix = pd.merge(matrix, items, on=['item_id'], how='left')
# interaction features: 

# - is item new

# - has it been bought in shop before



first_item_block = matrix.groupby(['item_id'])['date_block_num'].min().reset_index()

first_item_block['item_first_interaction'] = 1



first_shop_item_buy_block = matrix[matrix['date_block_num'] > 0].groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()

first_shop_item_buy_block['first_date_block_num'] = first_shop_item_buy_block['date_block_num']



matrix = pd.merge(matrix, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')

matrix = pd.merge(matrix, first_shop_item_buy_block[['item_id', 'shop_id', 'first_date_block_num']], on=['item_id', 'shop_id'], how='left')



matrix['first_date_block_num'].fillna(100, inplace=True)

matrix['shop_item_sold_before'] = (matrix['first_date_block_num'] < matrix['date_block_num']).astype('int8')

matrix.drop(['first_date_block_num'], axis=1, inplace=True)



matrix['item_first_interaction'].fillna(0, inplace=True)

matrix['shop_item_sold_before'].fillna(0, inplace=True)

 

matrix['item_first_interaction'] = matrix['item_first_interaction'].astype('int8')  

matrix['shop_item_sold_before'] = matrix['shop_item_sold_before'].astype('int8') 
matrix.head()
# remove first 12 months and split into train, validation and test data

matrix = matrix[matrix["date_block_num"] >= 12]



X_train = matrix[matrix["date_block_num"] < 33].drop(["item_cnt_month"], axis=1)

X_val = matrix[matrix["date_block_num"] == 33].drop(["item_cnt_month"], axis=1)

X_test = matrix[matrix["date_block_num"] == 34].drop(["item_cnt_month"], axis=1)



y_train = matrix[matrix["date_block_num"] < 33]["item_cnt_month"]

y_val = matrix[matrix["date_block_num"] == 33]["item_cnt_month"]
# quick linear regression analysis on validation data



results = sm.OLS(y_val.to_numpy(), X_val.astype(float)).fit()

results.summary()
%%time

xgb_model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



xgb_model.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, y_train), (X_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 10)



pickle.dump(xgb_model, open("xgb_model.p", "wb"))
lgbm_model = LGBMRegressor(

    n_estimators=200,

    learning_rate=0.03,

    num_leaves=32,

    colsample_bytree=0.9,

    subsample=0.8,

    max_depth=8,

    reg_alpha=0.04,

    reg_lambda=0.07,

    min_split_gain=0.02,

    min_child_weight=40,

    seed=42

)



lgbm_model.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, y_train), (X_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 10)



pickle.dump(lgbm_model, open("lgbm_model.p", "wb"))
# xgb feature importance

cols = X_val.columns

plt.figure(figsize=(10,10))

plt.barh(cols, xgb_model.feature_importances_)

plt.show()
# lgbm feature importance



cols = X_val.columns

plt.figure(figsize=(10,10))

plt.barh(cols, lgbm_model.feature_importances_)

plt.show()
# predict and combine using a linear combination of results



xgb_preds = xgb_model.predict(X_test)

lgbm_preds = lgbm_model.predict(X_test)



final_preds = (xgb_preds + lgbm_preds) / 2

final_preds = final_preds.clip(0,20)

final_preds
submission = pd.DataFrame({"item_cnt_month": final_preds})

submission.index.name="ID"

submission

submission.to_csv("submission.csv")