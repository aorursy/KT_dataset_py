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
# import the basic libraries we will use in this kernel
import os
import numpy as np
import pandas as pd
import pickle

import time
import datetime
from datetime import datetime
import calendar

from sklearn import metrics
from math import sqrt
import gc

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.preprocessing import LabelEncoder

import itertools
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore") # specify to ignore warning messages
# Resample the sales by this parameter
PERIOD = "M"

SHOPS = [8, 14, 37, 41, 59]

# this is help us change faster between Kaggle and local machine
LOCAL = False

if LOCAL:
    PATH = os.getcwd()
    FULL_DF_PATH = PATH
    GB_DF_PATH = PATH
    OUTPUT_PATH = PATH
else:
    PATH = '../input/competitive-data-science-predict-future-sales/'
    FULL_DF_PATH = "../input/full-df-only-test-all-features/"
    GB_DF_PATH = "../input/group-by-df/"
# prints the local files
def print_files():
    
    '''
    Prints the files that are in the current working directory.
    '''
    
    cwd = "../input/competitive-data-science-predict-future-sales/"
    
    for f, ff, fff in os.walk(cwd):
        for file in fff:
            if file.split(".")[1] in ["pkl", "csv"]:
                print(file)
# reduces the memory of a dataframe
def reduce_mem_usage(df, verbose = True):
    
    '''
    Reduces the space that a DataFrame occupies in memory.

    This function iterates over all columns in a df and downcasts them to lower type to save memory.
    '''
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'\
              .format(end_mem, 100 * (start_mem - end_mem) / start_mem))
def generate_shift_features(dataset, lista):
    objective_column_name = "_".join(lista)
    lista2=lista.copy()
    gb_df_=dataset.pivot_table(index=lista, values='item_cnt_day', aggfunc=[np.sum,np.mean,np.max]).reset_index()
    lista2.append(objective_column_name+'_sum')
    lista2.append(objective_column_name+'_mean')
    lista2.append(objective_column_name+'_max')
    gb_df_.columns=lista2    
    for tipo in ['_sum','_mean','_max']:
        # Para las agrupaciones de item_cnt_day
        # Shift 1,2,3,6
        for x in [1,2,3,4,6]:
            gb_df_[objective_column_name+tipo+'_shift_'+str(x)]=gb_df_.groupby(lista[1:])[objective_column_name+tipo].shift(x)

        # notice taht in var_3 we use shift(4), we do this because we want to capture the variation of 3 months
        # and not the variation of month - 3
        for x in [1,2,3]:
            gb_df_[objective_column_name+tipo+'_var_'+str(x)] = gb_df_[objective_column_name+tipo+'_shift_1']-\
                                                                gb_df_[objective_column_name+tipo+'_shift_'+str(x+1)]
        for x in [1,2,3]:
             gb_df_[objective_column_name+tipo+'_var_pct_'+str(x)] = gb_df_[objective_column_name+tipo+'_var_'+str(x)]/\
                                                                    gb_df_[objective_column_name+tipo+'_shift_'+str(x+1)]
    gb_df_.fillna(-2, inplace = True)  
    gb_df_.replace([np.inf, -np.inf], -2, inplace = True)
    return gb_df_
file_dir="/kaggle/input/future-sales-part-ii-model-training-2-modelo/FULL_DF_ANTES_DEL_FEATUREENGENIRERING.pkl"
full_df = pd.read_pickle(file_dir)
gb_list = ["date", "shop_id", "city"]
shop_sales_features =  generate_shift_features(full_df, gb_list)
shop_sales_features[shop_sales_features['shop_id']==8 ].head().T
gb_list = ["date", "item_id"]
st = time.time()
item_sales_features =  generate_shift_features(full_df, gb_list)
et = time.time()
(et - st)/60
item_sales_features[item_sales_features["item_id"] == 30 ].head().T
gb_list = ["date", "item_category_id"]
month_item_category_features =  generate_shift_features(full_df, gb_list)
month_item_category_features[month_item_category_features["item_category_id"] == 2].head().T
gb_list = ["date", "type_code"]
month_type_code_features =  generate_shift_features(full_df, gb_list)

month_type_code_features[month_type_code_features["type_code"] == 1].head().T
full_df["year"] = full_df["date"].dt.year
full_df["month"] = full_df["date"].dt.month
full_df["days_in_month"] = full_df["date"].dt.days_in_month
full_df["quarter_start"] = full_df["date"].dt.is_quarter_start
full_df["quarter_end"] = full_df["date"].dt.is_quarter_end
full_df.head()
holidays_next_month = {
    12:8,
    1:1,
    2:1,
    3:0,
    4:2,
    5:1,
    6:0,
    7:0,
    8:0,
    9:0,
    10:1,
    11:0
}

holidays_this_month = {
    1:8,
    2:1,
    3:1,
    4:0,
    5:2,
    6:1,
    7:0,
    8:0,
    9:0,
    10:0,
    11:1,
    12:0
}

full_df["holidays_next_month"] = full_df["month"].map(holidays_next_month)
full_df["holidays_this_month"] = full_df["month"].map(holidays_this_month)
def extract_number_weekends(test_month):
    saturdays = len([1 for i in calendar.monthcalendar(test_month.year, test_month.month) if i[5] != 0])
    sundays = len([1 for i in calendar.monthcalendar(test_month.year, test_month.month) if i[6] != 0])
    return saturdays + sundays

full_df["total_weekend_days"] = full_df["date"].apply(extract_number_weekends)

date_diff_df = full_df[full_df["item_cnt_day"] > 0][["shop_id", "item_id", "date", "item_cnt_day"]].groupby(["shop_id", "item_id"])\
["date"].diff().apply(lambda timedelta_: timedelta_.days).to_frame()

date_diff_df.columns = ["date_diff_sales"]

full_df = pd.merge(full_df, date_diff_df, how = "left", left_index=True, right_index=True)

full_df.fillna(-1, inplace = True)
full_df.head()
city_population = {\
'Якутск':307911, 
'Адыгея':141970,
'Балашиха':450771, 
'Волжский':326055, 
'Вологда':313012, 
'Воронеж':1047549,
'Выездная':1228680, 
'Жуковский':107560, 
'Интернет-магазин':1228680, 
'Казань':1257391, 
'Калуга':341892,
'Коломна':140129,
'Красноярск':1083865, 
'Курск':452976, 
'Москва':12678079,
'Мытищи':205397, 
'Н.Новгород':1252236,
'Новосибирск':1602915 , 
'Омск':1178391, 
'РостовНаДону':1125299, 
'СПб':5398064, 
'Самара':1156659,
'СергиевПосад':104579, 
'Сургут':373940, 
'Томск':572740, 
'Тюмень':744554, 
'Уфа':1115560, 
'Химки':244668,
'Цифровой':1228680, 
'Чехов':70548, 
'Ярославль':608353
}

city_income = {\
'Якутск':70969, 
'Адыгея':28842,
'Балашиха':54122, 
'Волжский':31666, 
'Вологда':38201, 
'Воронеж':32504,
'Выездная':46158, 
'Жуковский':54122, 
'Интернет-магазин':46158, 
'Казань':36139, 
'Калуга':39776,
'Коломна':54122,
'Красноярск':48831, 
'Курск':31391, 
'Москва':91368,
'Мытищи':54122, 
'Н.Новгород':31210,
'Новосибирск':37014 , 
'Омск':34294, 
'РостовНаДону':32067, 
'СПб':61536, 
'Самара':35218,
'СергиевПосад':54122, 
'Сургут':73780, 
'Томск':43235, 
'Тюмень':72227, 
'Уфа':35257, 
'Химки':54122,
'Цифровой':46158, 
'Чехов':54122, 
'Ярославль':34675
}

full_df["city_population"] = full_df["city"].map(city_population)

full_df["city_income"] = full_df["city"].map(city_income)

full_df["price_over_income"] = full_df["item_price"]/full_df["city_income"]
full_df = pd.merge(full_df, shop_sales_features, on = ["date", "shop_id"], how = "left")


full_df = pd.merge(full_df, item_sales_features, on = ["date", "item_id"], how = "left")


full_df = pd.merge(full_df, month_item_category_features, on = ["date", "item_category_id"], how = "left")


full_df = pd.merge(full_df, month_type_code_features, on = ["date", "type_code"], how = "left")
list(full_df.head().columns)
# delete dfs with features
del shop_sales_features, item_sales_features, month_item_category_features, month_type_code_features
# delete all the previous df
#del shops_df, items_df, items_category_df, sales_df, test_df, cartesian_product, gb_df
gc.collect()
full_df.rename(columns = {"item_cnt_day":"sales"}, inplace = True)
full_df['sales'].max()
st = time.time()

full_df.to_pickle("FULL_DF_ONLY_TEST_ALL_FEATURES.pkl")

et = time.time()

(et - st)/60
# Solom para 5 tiendas
full_df = full_df[full_df["shop_id"].isin(SHOPS)]
# delete all the columns where lags features are - 2 (shift(6))
full_df = full_df[full_df["date"] > np.datetime64("2013-06-30")]
cols_to_drop = [

'revenue',
'shop_name',
'city_x',
'city_y',
'item_name',
'item_category_name',
'split',
'type',
'subtype',
    
"date_shop_id_city_sum",
"date_shop_id_city_mean",
"date_shop_id_city_max",

"date_item_id_sum",
"date_item_id_mean",
"date_item_id_max",
    
"date_item_category_id_sum",
"date_item_category_id_mean",
"date_item_category_id_max",

"date_type_code_sum",
"date_type_code_mean",
"date_type_code_max"]
full_df.drop(cols_to_drop, inplace = True, axis = 1)
# split the data into train, validation and test dataset
train_index = sorted(list(full_df["date"].unique()))[:-2]

valida_index = [sorted(list(full_df["date"].unique()))[-2]]

test_index = [sorted(list(full_df["date"].unique()))[-1]]
X_train = full_df[full_df["date"].isin(train_index)].drop(['sales', "date"], axis=1)
Y_train = full_df[full_df["date"].isin(train_index)]['sales']

X_valida = full_df[full_df["date"].isin(valida_index)].drop(['sales', "date"], axis=1)
Y_valida = full_df[full_df["date"].isin(valida_index)]['sales']

X_test = full_df[full_df["date"].isin(test_index)].drop(['sales', "date"], axis = 1)
Y_test = full_df[full_df["date"].isin(test_index)]['sales']
st = time.time()

model = XGBRegressor(seed = 175)

model_name = str(model).split("(")[0]

day = str(datetime.now()).split()[0].replace("-", "_")
hour = str(datetime.now()).split()[1].replace(":", "_").split(".")[0]
t = str(day) + "_" + str(hour)

model.fit(X_train, Y_train, eval_metric = "rmse", 
    eval_set = [(X_train, Y_train), (X_valida, Y_valida)], 
    verbose = True, 
    early_stopping_rounds = 10)

et = time.time()

print("Training took {} minutes!".format((et - st)/60))
pickle.dump(model, open("{}_{}.dat".format(model_name, t), "wb"))
print("{}_{}.dat".format(model_name, t))
importance = model.get_booster().get_score(importance_type = "gain")

importance = {k: v for k, v in sorted(importance.items(), key = lambda item: item[1])}
fig, ax = plt.subplots(figsize=(15, 30))
plot_importance(model, importance_type = "gain", ax = ax)
plt.savefig("{}_{}_plot_importance.png".format(model_name, t))
Y_valida_pred = model.predict(X_valida)
metrics.r2_score(Y_valida, Y_valida_pred)
rmse_valida = sqrt(metrics.mean_squared_error(Y_valida, Y_valida_pred))
rmse_valida
Y_test_predict = model.predict(X_test)
Y_test_predict.sum()
Y_test_predict.max()
Y_test.sum()
Y_test.max()
