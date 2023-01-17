###   load the required libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt



###   set some configurations for plots

plt.rcParams['figure.figsize'] = (24,18)

pd.set_option('display.max_columns', None)
import os

print(os.listdir("../input"))
###   read in the data files

item_cat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sales_test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

df = sales_train.copy()

df.dtypes
###   take a look at a sample of the files

print(item_cat.head())

print(items.head())

print(shops.head())

print(sales_train.head())

print(sales_train.tail())

print(sales_test.head())
# convert 'date' column into type "datetime"

from datetime import datetime

df.date = pd.to_datetime(df.date, format = "%d.%m.%Y")

df.dtypes

print(df.head(10))
def create_period(df):

    # create a function to obtain the year and month info (for time series)		

    get_mth = lambda x: x.strftime('%m')	

    df['mth'] = df.date.apply(get_mth)

    df.mth = df.mth.astype('int64')



    #get_yr_mth = lambda x: x.strftime('%Y-%m')

    #get_yr = lambda x: x.strftime('%Y')

    #df['yr_mth'] = df.date.apply(get_yr_mth)

    #df['yr'] = df.date.apply(get_yr)

    #df.yr.astype('datetime64[M]')

    return df

# create month info 

df_period = create_period(df)

df_period.shape

print(df_period.head())
pivot_df = df_period.pivot_table(index = ['shop_id', 'item_id', 'mth'], 

								 values = 'item_cnt_day',

								 columns = 'date_block_num',

								 aggfunc = 'sum').fillna(0.0).reset_index()



pivot_df = pivot_df.groupby(['shop_id','item_id']).max().reset_index() ###

pivot_df.head()



pivot_df.shape

df2 = pivot_df.reset_index()

df2.head()

df2.shape

###   merge in all the relevant info into the earlier "df" for completeness and for ML

df3 = pd.merge(df2, items, how = "inner", on = "item_id")

df4 = pd.merge(df3, item_cat, how = "inner", on = "item_category_id")

df5 = pd.merge(df4, shops, how = "inner", on = "shop_id")

df5.head()

df5.shape
df5.isna().sum() # there are no NAs
# we change the order of the columns

[(i, df5.columns[i]) for i in range(len(df5.columns))]

col_order = [0,1,41,2,38,39,40,3] + list(range(4,38))

df6 = df5.iloc[:,col_order]

df6.head()

###   create some visualisations

df6.hist()
###############################################################################

###   create a machine learning model

###############################################################################



# using all data to predict 'item_price' given that 'date_block_num' is 34

# we insert the price info as a row, mth as 11, 

# then use the predicted item price to predict the item count for the month

X = df6.loc[:,df6.columns != 33]

X = X.drop(['item_name', 'item_category_name', 'shop_name'], axis = 1)

y = df5.loc[:,df5.columns == 33]

print(X.head())

print(y.head())



# create an XGBoost model

import xgboost as xgb

data_dmatrix = xgb.DMatrix(data = X, label = y)



# do a train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

xg_reg = xgb.XGBRegressor(num_round = 1000, 

                          verbosity = 0, # silent all warning messages                          

                          eval_metric = 'rmse',                          

                          min_child_weight = 0.5, 

                          eta = 0.2, # something like learning rate

                          seed = 100,                          

                          gamma = 10, # min loss reduction to make a further partition

                          max_depth = 8, # increasing this value makes model more complex

                          n_estimators = 100)



xg_reg.fit(X_train, y_train)
y_preds = xg_reg.predict(X_test)

print(y_preds)
# we measure the accuracy using RMSE

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, pd.DataFrame(y_preds)))

print(rmse)
###############################################################################

###   do the final prediction

###############################################################################



# this is based on the merged data with the 'sales_test' dataframe

test_df = sales_test.merge(df6, how = 'left', on = ['shop_id', 'item_id']).fillna(0.0)



# reorder the columns again, and dropping less important columns

[(i,test_df.columns[i]) for i in range(len(test_df.columns))]

col_order_test = [3,1,2,6,8] + list(range(9,43))

test_df = test_df.iloc[:,col_order_test]



# previously we fitted the model without one column, 

# so, now we want to predict the 34th column, so we move one month forward

# i.e. from 1 to 33 (including renaming the column names)

dic_names = dict(zip(test_df.columns[5:], list(np.array(list(test_df.columns[5:])) - 1)))

test_df = test_df.rename(dic_names, axis = 1)



test_df_select = test_df.iloc[:,test_df.columns != -1]

test_df_select.head()



# the final prediction

y_preds_test = xg_reg.predict(test_df_select)
# we clip the predictions to a range between 0 to 20

clipped_preds = list(map(lambda x: min(20, max(x,0)), list(y_preds_test)))

results = pd.DataFrame({'ID': test_df.index, 'item_cnt_month': clipped_preds})



results.head(20)

results.shape
# read into the submission file

results.describe()

results.to_csv('try.csv')