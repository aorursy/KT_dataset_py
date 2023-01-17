# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from tqdm import tqdm_notebook as tqdm

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print("$$")
# Changes a name of a column in a DataFrame

def change_col_name(col, old_name, new_name):

    return col if col != old_name else new_name





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

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    

    return df



def downcast_dtypes16(df):

    '''

        Changes column types in the dataframe: 

                

                `float32` type to `float16`

                `int32`   type to `int16`

    '''

    

    # Select columns to downcast

    float_cols = [c for c in df if df[c].dtype == "float32"]

    int_cols =   [c for c in df if df[c].dtype == "int32"]

    

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float16)

    df[int_cols]   = df[int_cols].astype(np.int16)

    

    return df
sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

sales_test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
print(sales_train.shape)

print(sales_test.shape)

sales_train.head(10)

sales_test.head(10)
shops.head(10)
items.head(10)
sample_submission.head(100)
print(sales_train.isnull().sum(axis=1))

print(sales_train.isnull().sum(axis=1).sum())

print(sales_train.isnull().sum(axis=0).head())
feats_values = sales_train.nunique(dropna=False)

feats_values.sort_values()[:10]
print(sales_train["item_cnt_day"].nunique())

print(sales_train["item_cnt_day"].value_counts().head(20))

print(sales_train["item_cnt_day"].value_counts().tail(20))

print(sales_train["item_cnt_day"].sort_values(ascending=False).head(10))
plt.figure(figsize=(14,6))

_ = plt.hist(sales_train["item_cnt_day"], bins=1000)

plt.xlim([0,15])
sales_train["sale_value"] = sales_train["item_cnt_day"] * sales_train["item_price"]

sales_train.head(10)
plt.figure(figsize=(16,8))

sns.boxplot(sales_train["item_cnt_day"])

plt.show()
plt.figure(figsize=(16,8))

sns.boxplot(sales_train["item_price"])

plt.show()
sales_train = sales_train[sales_train.item_price < 100000]

sales_train = sales_train[sales_train.item_cnt_day < 1001]
shop_sales = sales_train.loc[:,["shop_id", "sale_value", "item_cnt_day"]].groupby(["shop_id"]).sum().sort_values(["shop_id"])

shop_sales.head(10)
shop_sales.sort_values(["sale_value"], ascending=False).head(10)
shop_sales.sort_values(["item_cnt_day"], ascending=False).head(10)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(17,6))

ax1.plot(shop_sales["sale_value"])

ax2.plot(shop_sales["item_cnt_day"])

ax1.set_title("Total Sales")

ax2.set_title("Number Of Sales")



from itertools import product



#def X_train

X_train = sales_train.groupby(["date_block_num", "shop_id", "item_id"]).sum().drop(["item_price"], axis=1).reset_index()



#Concatinate test and train

sales_test["date_block_num"] = 34

test = sales_test.drop(["ID"], axis=1)

X_train = pd.concat([X_train, test], axis=0)

X_train = X_train.fillna(0)

#X_train = X_train.loc[X_train["shop_id"].isin([25,26,27,28])]

# Create "grid" with columns

index_cols = ['shop_id', 'item_id', 'date_block_num']



# For every month we create a grid from all shops/items combinations from that month

grid = [] 

for block_num in X_train['date_block_num'].unique():

    cur_shops = X_train.loc[X_train['date_block_num'] == block_num, 'shop_id'].unique()

    cur_items = X_train.loc[X_train['date_block_num'] == block_num, 'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



# Turn the grid into a dataframe

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)



# Groupby data to get shop-item-month aggregates



gb = sales_train.groupby(index_cols,as_index=False).agg({'item_cnt_day': 'sum', 'sale_value': 'sum'})

# Fix column names

gb.columns = [change_col_name(col, 'item_cnt_day', 'target') for col in gb.columns]

gb.columns = [change_col_name(col, 'sale_value', 'sale_value_month') for col in gb.columns]

# Join it to the grid

X_train = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)



# Get shop-month agg

gb = sales_train.groupby(["date_block_num", "shop_id"]).agg({"item_cnt_day": "sum", "sale_value": 'sum'})

# Fix column names

gb.columns = [change_col_name(col, 'item_cnt_day', 'shop_month_item_sale') for col in gb.columns]

gb.columns = [change_col_name(col, 'sale_value', 'shop_sale_value_month') for col in gb.columns]

# Join it to the grid

X_train = pd.merge(X_train, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)



#### clip target norm because of competition rules ####

X_train["target"] = X_train["target"].clip(0,20)



# Get item-month agg

gb = sales_train.groupby(["date_block_num", "item_id"]).agg({"item_cnt_day": "sum", "sale_value": 'sum'})

# Fix column names

gb.columns = [change_col_name(col, 'item_cnt_day', 'item_month_item_sale') for col in gb.columns]

gb.columns = [change_col_name(col, 'sale_value', 'item_sale_value_month') for col in gb.columns]

# Join it to the grid

X_train = pd.merge(X_train, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)



# Get item-month avg

gb = sales_train.groupby(["date_block_num", "item_id"]).agg({"item_cnt_day": "mean", "sale_value": 'mean'})

# Fix column names

gb.columns = [change_col_name(col, 'item_cnt_day', 'item_month_avg_sale') for col in gb.columns]

gb.columns = [change_col_name(col, 'sale_value', 'item_sale_value_avg_month') for col in gb.columns]

# Join it to the grid

X_train = pd.merge(X_train, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)



# Get shop-month avg

gb = sales_train.groupby(["date_block_num", "shop_id"]).agg({"item_cnt_day": "mean", "sale_value": 'mean'})

# Fix column names

gb.columns = [change_col_name(col, 'item_cnt_day', 'shop_month_item_avg') for col in gb.columns]

gb.columns = [change_col_name(col, 'sale_value', 'shop_sale_value_avg') for col in gb.columns]

# Join it to the grid

X_train = pd.merge(X_train, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)



# Downcast dtypes

X_train = downcast_dtypes(X_train)

X_train

dates = X_train["date_block_num"]

shift_range = [1, 2, 3, 4, 5, 12]



def delay_change(col, delay):

    return "{}_lag_{}".format(col, delay)

cols_to_delay = list(set(X_train.columns) - set(index_cols))

for shift in tqdm(shift_range):

    shifted = X_train[cols_to_delay + index_cols]

    shifted["date_block_num"] = dates + 1

    foo = lambda x: '{}_lag_{}'.format(x, str(shift)) if x in cols_to_delay else x

    shifted = shifted.rename(columns=foo)

    X_train =  X_train.merge(shifted, on=index_cols, how='left')



X_train = X_train.fillna(0)

# Discard old data

X_train = X_train.loc[dates >= 13]
X_train
# adding item category id

X_train = X_train.merge(items.loc[:,["item_id", "item_category_id"]], on=["item_id"], how='left').fillna(0)
feature_mat = X_train.drop(["target", "sale_value_month", "shop_month_item_sale", "item_sale_value_month",

                            "shop_sale_value_month", "item_month_item_sale", "shop_month_item_avg",

                            "shop_sale_value_avg", "item_month_avg_sale", "item_sale_value_avg_month"], axis=1)

target = X_train["target"]

X_train = feature_mat
X_train
good_features=[]

for col in X_train.columns:

    if(abs(X_train.loc[:,col].corr(target) > 0.4)):

        good_features.append(col)

print(good_features)
#plotting feature correlation

print(len(good_features))

print(len(X_train.columns))

cor = X_train.loc[:,good_features].corr()

plt.figure()

sns.heatmap(cor)

plt.show()



from sklearn.ensemble import RandomForestRegressor

#r_forest_reg = RandomForestRegressor(n_jobs=4, verbose=1)

#r_forest_reg.fit(X_train, target[dates < 34])
#pd.Series(r_forest_reg.feature_importances_).to_csv("/kaggle/working/feature_imp.csv")
rf_feature_importance = pd.read_csv("/kaggle/input/rf-features/feature_imp.csv") # load_data

#rf_feature_importance = pd.Series(r_forest_reg.feature_importances_)

rf_feature_importance.columns = ["feature_num", "feature_imp"]

plt.figure(figsize=(14,8))

plt.plot(rf_feature_importance.iloc[:,1:])

plt.show()
important_10 = rf_feature_importance.nlargest(10, ["feature_imp"])

important_10 = X_train.columns[important_10.index] # extract feature names

important_10
relevant_features = set((*good_features, *important_10))

print(len(relevant_features))

print(relevant_features)
dates = feature_mat["date_block_num"]

X_test = feature_mat.loc[dates == 34]

Y_train = target.loc[dates <=32]

X_train = feature_mat[dates <= 32]

X_cv = feature_mat.loc[dates == 33]

Y_cv = target.loc[dates == 33]

# Save data to disk

#X_train.to_csv("/kaggle/working/X_train.csv")

#X_test.to_csv("/kaggle/working/X_test.csv")

#Y_train.to_csv("/kaggle/working/Y_train.csv")
import lightgbm as lgb

lgb_params = {

               'feature_fraction': 0.75,

               'metric': 'rmse',

               'nthread':4, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.75, 

               'learning_rate': 0.05, 

               'objective': 'mse', 

               'bagging_seed': 2**7, 

               'num_leaves': 2**7,

               'bagging_freq':1,

               'verbose':0,

               'early_stopping_rounds': 10,

               'verbose_eval': True

              }



model = lgb.train(lgb_params, lgb.Dataset(X_train, label=Y_train), 200, valid_sets=lgb.Dataset(X_cv, Y_cv))

pred_lgb = model.predict(X_test)
pred = pred_lgb

print(pred)

submission = pd.DataFrame()

submission["ID"] = range(len(pred))

submission["item_cnt_month"] = pred

submission = submission.set_index(['ID'], drop=True)
pred.max()
submission.to_csv("/kaggle/working/submission.csv")
lgb.plot_importance(model, figsize=(16,8))


gbm_feats = feature_mat.columns[np.argsort(-model.feature_importance())[:25]]

relevant_features = relevant_features.union(gbm_feats)

final_feat_mat = feature_mat.loc[:,relevant_features]
#final_feat_mat.to_json("/kaggle/working/feature_mat.jsn")

#target.to_csv("/kaggle/working/target.csv")

#model.save_model("/kaggle/working/raw_lgbm_model.txt")

import gc

del feature_mat

del X_train

del X_cv

gc.collect()
from sklearn.metrics import mean_squared_error as mse

dates = final_feat_mat["date_block_num"]

X_train = final_feat_mat[dates <= 32]

X_cv = final_feat_mat[dates == 33]
# args: linear model, random forest model, gbdt model, categorical feature indices. output: models r2 scroes

def cross_valid_scores(l_reg, r_forest_reg, gbdt,return_level2=False):

    r_forest_score = 0

    l_reg_score = 0

    lgb_score = 0

    if return_level2:

        level2 = np.zeros([dates_level2.shape[0], 3])

    for date_block in tqdm([29,30,31,32,33]):

        X_train_short = X_train[dates < date_block]

        Y_train_short = Y_train.loc[dates < date_block]

        X_cv = X_train[dates == date_block]

        Y_cv = Y_train.loc[dates == date_block]

        # Fit models

        r_forest_reg.fit(X_train_short, Y_train_short)

        l_reg.fit(X_train_short, Y_train_short)

        gbdt = lgb.train(lgb_params, lgb.Dataset(X_train_short, label=Y_train_short), 200, valid_sets=lgb.Dataset(X_cv, Y_cv))



        # Predict

        l_reg_pred = l_reg.predict(X_cv)

        r_forest_pred = r_forest_reg.predict(X_cv)

        lgb_pred = gbdt.predict(X_cv)

            

        # Eval scores

        r_forest_score += np.sqrt(mse(Y_cv, r_forest_pred))

        l_reg_score += np.sqrt(mse(Y_cv, l_reg_pred))

        lgb_score += np.sqrt(mse(Y_cv, lgb_pred))

        print(np.array([r_forest_score, l_reg_score, lgb_score]))

        

        #fill level2 table

        if return_level2:

            level2[dates_level2 == date_block, 0] = r_forest_pred

            level2[dates_level2 == date_block, 1] = l_reg_pred

            level2[dates_level2 == date_block, 2] = lgb_pred

            

    # Get avg prediction scores

    score = np.array([r_forest_score, l_reg_score, lgb_score])

    score = score / 5

    if return_level2:

        return level2

    else:

        return score
# defining parameters

rf_params = {

    'n_estimators': 20,

    'max_depth': 20,

    'n_jobs': 4

}

rf = RandomForestRegressor(**rf_params)

rf.fit(X_train, Y_train)

rf_pred = rf.predict(X_cv)

rf_score = np.sqrt(mse(Y_cv, rf_pred))

print(rf_score)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# sampling shops

sample_shops = np.random.randint(0, high=(final_feat_mat["shop_id"].max()+1), size=8)

X_train = final_feat_mat[final_feat_mat["shop_id"].isin(sample_shops)]

dates = X_train["date_block_num"]

Y_train = target[final_feat_mat["shop_id"].isin(sample_shops)]



# define y for relevant month

Y_ensemble = Y_train.loc[dates.isin([29,30,31,32,33])]

dates_level2 = X_train.loc[X_train["date_block_num"].isin([29, 30,31,32,33]), "date_block_num"]

# evaluate model scores

level2_feats = cross_valid_scores(lr, rf, model,return_level2=True)



# train linear regressor

ensemble_model = LinearRegression()

ensemble_model.fit(level2_feats, Y_ensemble)
print(final_feat_mat.shape)

print(X_train.shape)

print(Y_train.shape)

print(Y_ensemble.shape)

print(X_train[X_train["date_block_num"].isin([29,30,31,32,33])].shape)

print(level2_feats.shape)
# creating final traning sets

dates = final_feat_mat["date_block_num"]

Y_train = target.loc[dates <=32]

X_train = final_feat_mat[dates <= 32]



Y_cv = target.loc[dates == 33]

X_cv = final_feat_mat[dates == 33]



# define test set

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

test["date_block_num"] = 34

test = test.set_index(['ID'], drop=True)

test = test.merge(final_feat_mat, on=['date_block_num', 'item_id', 'shop_id'], how='left')



# train models

cb = lgb.train(lgb_params, lgb.Dataset(X_train, label=Y_train), 200, valid_sets=lgb.Dataset(X_cv, Y_cv))

rf.fit(X_train, Y_train)

lr.fit(X_train, Y_train)



# predict for test set

pred_cb = cb.predict(test)

pred_rf = rf.predict(test)

pred_lr = lr.predict(test)



# ensemble predicts

final_prediction = ensemble_model.predict(np.c_[pred_rf, pred_lr, pred_cb])

submission = pd.DataFrame()

submission["ID"] = range(len(pred))

submission["item_cnt_month"] = final_prediction

submission = submission.set_index(['ID'], drop=True)
print(np.sqrt(mse(Y_cv, ensemble_model.predict(np.c_[rf.predict(X_cv), cb.predict(X_cv), lr.predict(X_cv)]))))

submission = submission.clip(lower=0)

submission.max()
submission.to_csv("/kaggle/working/sub.csv")