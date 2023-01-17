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
# SHOPS = [8, 11, 14, 25, 28, 31, 33, 36, 37, 40, 41, 42, 44, 51, 52, 54, 59]

SHOPS = [8, 14, 37, 41, 59]

# this is help us change faster between Kaggle and local machine
LOCAL = False

if LOCAL:
    PATH = os.getcwd()
else:
    PATH = '../input/competitive-data-science-predict-future-sales/'
    PATH_EXTERNAL = "../input/group-by-df/"
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
print_files()
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
# load all the df we have
shops_df = pd.read_csv(os.path.join(PATH, "shops.csv"))
items_df = pd.read_csv(os.path.join(PATH, "items.csv"))
items_category_df = pd.read_csv(os.path.join(PATH, "item_categories.csv"))
sales_df = pd.read_csv(os.path.join(PATH, "sales_train.csv"))
test_df = pd.read_csv(os.path.join(PATH, "test.csv"))
# sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))
# we have seen in our EDA that we have some duplicate shops, let's correct them.
shops_df.loc[shops_df.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops_df['city'] = shops_df['shop_name'].str.split(' ').map(lambda x: x[0])
shops_df.loc[shops_df.city == '!Якутск', 'city'] = 'Якутск'
shops_df['city_code'] = LabelEncoder().fit_transform(shops_df['city'])
shops_df.head()
shops_df[shops_df["shop_id"].isin([0, 57])]
# Якутск Орджоникидзе, 56
sales_df.loc[sales_df.shop_id == 0, 'shop_id'] = 57
test_df.loc[test_df.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"
sales_df.loc[sales_df.shop_id == 1, 'shop_id'] = 58
test_df.loc[test_df.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²
sales_df.loc[sales_df.shop_id == 10, 'shop_id'] = 11
test_df.loc[test_df.shop_id == 10, 'shop_id'] = 11
items_category_df['split'] = items_category_df['item_category_name'].str.split('-')
items_category_df['type'] = items_category_df['split'].map(lambda x: x[0].strip())
items_category_df['type_code'] = LabelEncoder().fit_transform(items_category_df['type'])

# if subtype is nan then type
items_category_df['subtype'] = items_category_df['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
items_category_df['subtype_code'] = LabelEncoder().fit_transform(items_category_df['subtype'])

items_category_df.head()
sales_df.head()
# we have negative prices and some outlier
# let's replace the data with the mean value and also filter all the outliers
mean = sales_df[(sales_df["shop_id"] == 32) & (sales_df["item_id"] == 2973) & (sales_df["date_block_num"] == 4) & (sales_df["item_price"] > 0)]["item_price"].mean()
sales_df.loc[sales_df.item_price < 0, 'item_price'] = mean

sales_df = sales_df[sales_df["item_price"] < np.percentile(sales_df["item_price"], q = 100)]
sales_df = sales_df[sales_df["item_cnt_day"] < np.percentile(sales_df["item_cnt_day"], q = 100)]
sales_df.info()
type(sales_df["date"].iloc[0])
# convert to datetime the date column
# specify the format since otherwise it might give some problems
sales_df["date"] = pd.to_datetime(sales_df["date"], format = "%d.%m.%Y")
# max date in sales is 31.10.2015.
# In the Kaggle competition we are asked to predict the sales for the next month
# this means the sales of November
min_date = sales_df["date"].min()
max_date_sales = sales_df["date"].max()
max_date_sales
max_date_test = datetime(2015, 11, 30)
date_range = pd.date_range(min_date, max_date_test, freq = "D")
date_range
len(date_range)
shops = sorted(list(shops_df["shop_id"].unique()))

# only items present in test
items = sorted(list(items_df["item_id"].unique()))

cartesian_product = pd.MultiIndex.from_product([date_range, shops, items], names=["date", "shop_id", "item_id"])
len(cartesian_product)
date_range = pd.date_range(min_date, max_date_test, freq = "W")
date_range
len(date_range)
shops = sorted(list(shops_df["shop_id"].unique()))

# only items present in test
items = sorted(list(items_df[items_df["item_id"].isin(test_df["item_id"].unique())]["item_id"].unique()))

cartesian_product = pd.MultiIndex.from_product([date_range, shops, items], names=["date", "shop_id", "item_id"])
len(cartesian_product)
date_range = pd.date_range(min_date, max_date_sales, freq = PERIOD)
date_range
len(date_range)
# only items present in test
items = sorted(list(test_df["item_id"].unique()))

cartesian_product = pd.MultiIndex.from_product([date_range, SHOPS, items], names = ["date", "shop_id", "item_id"])
len(cartesian_product)
gc.collect()
# st = time.time()

# # set index
# sales_df["revenue"] = sales_df["item_cnt_day"]*sales_df["item_price"]
# gb_df = sales_df.set_index("date")

# # groupby shop_id and item_id
# gb_df = gb_df.groupby(["shop_id", "item_id"])

# # resample the sales to a weekly basis
# gb_df = gb_df.resample(PERIOD).agg({'item_cnt_day': np.sum, "item_price": np.mean, "revenue":np.sum})

# # convert to dataframe and save the full dataframe
# gb_df.reset_index(inplace = True)

# # save the groupby dataframe
# gb_df.to_pickle(os.path.join(PATH, "GROUP_BY_DF.pkl"))

# et = time.time()

# print("Total time in minutes to preprocess took {}".format((et - st)/60))

# read the groupby dataframe
gb_df = pd.read_pickle(os.path.join(PATH_EXTERNAL, "GROUP_BY_DF.pkl"))
gb_df.head()
gb_df.isnull().sum()
gb_df.fillna(0, inplace = True)
full_df = pd.DataFrame(index = cartesian_product).reset_index()
full_df = pd.merge(full_df, gb_df, on = ['date','shop_id', "item_id"], how = 'left')

full_df["item_cnt_day"].sum()
full_df.shape
full_df.head()
# add shops_df information
full_df = pd.merge(full_df, shops_df, on = "shop_id")
full_df.head()
# add items_df information
full_df = pd.merge(full_df, items_df, on = "item_id")
full_df.head()
# add items_category_df information
full_df = pd.merge(full_df, items_category_df, on = "item_category_id")
full_df.head()
full_df.isnull().sum()
full_df.fillna(0, inplace = True)
# We will clip the value in this line.
# This means that the values greater than 20, will become 20 and lesser than 20
full_df["item_cnt_day"] = np.clip(full_df["item_cnt_day"], 0, 20)
# reduce_mem_usage(full_df)
full_df.head()
class FeatureGenerator(object):
    
    '''
    This is a helper class that takes a df and a list of features and creates sum, mean, 
    lag features and variation (change over month) features.
    
    '''
    
    def __init__(self, full_df,  gb_list):
        
        '''
        Constructor of the class.
        gb_list is a list of columns that must be in full_df.
        '''
        
        self.full_df = full_df
        self.gb_list = gb_list
        # joins the gb_list, this way we can dinamically create new columns
        # ["date, "shop_id] --> date_shop_id
        self.objective_column_name = "_".join(gb_list)

    @staticmethod
    def reduce_mem_usage(df, verbose = True):
        
        '''
        Reduces the space that a DataFrame occupies in memory.
        This is a static method of FeatureGenerator class (we can use it outside the class).
        
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
            
    def generate_gb_df(self):
        
        '''
        This function thakes the full_df and creates a groupby df based on the gb_list.
        It creates 2 columns: 
            1. A sum column for every date and gb_list
            2. Mean columns for every_date and gb_list
            
        The resulting df (gb_df_) is assigned back to the FeatureGenerator class as an attribute.
        '''

        def my_agg(full_df_, args):
            
            '''
            This function is used to perform multiple operations over a groupby df and returns a df
            without multiindex.
            '''
            
            names = {
                '{}_sum'.format(args):  full_df_['item_cnt_day'].sum(),
                '{}_mean'.format(args): full_df_['item_cnt_day'].mean()
            }

            return pd.Series(names, index = [key for key in names.keys()])
        
        # the args is used to pass additional argument to the apply function
        gb_df_ = self.full_df.groupby(self.gb_list).apply(my_agg, args = (self.objective_column_name)).reset_index()

        self.gb_df_ = gb_df_

        
    def return_gb_df(self):  
        
        '''
        This function takes the gb_df_ created in the previous step (generate_gb_df) and creates additional features.
        We create 4 lag features (values from the past).
        And 6 variation features: 3 with absolute values and 3 with porcentual change.
        '''
        
        def generate_shift_features(self, suffix):
            
            '''
            This function is a helper function that takes the gb_df_ and a suffix (sum or mean) and creates the
            additional features.
            '''

            # dinamically creates the features
            # date_shop_id --> date_shop_id_sum if suffix is sum
            # date_shop_id --> date_shop_id_mean if suffix is mean
            name_ = self.objective_column_name + "_" + suffix

            self.gb_df_['{}_shift_1'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(1))
            
            self.gb_df_['{}_shift_2'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(2))
            
            self.gb_df_['{}_shift_3'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(3))
            
            self.gb_df_['{}_shift_6'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(6))
            
            # notice taht in var_3 we use shift(4), we do this because we want to capture the variation of 3 months
            # and not the variation of month - 3

            self.gb_df_['{}_var_1'.format(name_)] = self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(1) - x.shift(2))
            self.gb_df_['{}_var_2'.format(name_)] = self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(1) - x.shift(3))
            self.gb_df_['{}_var_3'.format(name_)] = self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: x.shift(1) - x.shift(4))

            self.gb_df_['{}_var_pct_1'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: (x.shift(1) - x.shift(2))/x.shift(2))
            
            self.gb_df_['{}_var_pct_2'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: (x.shift(1) - x.shift(3))/x.shift(3))
            
            self.gb_df_['{}_var_pct_3'.format(name_)] =\
            self.gb_df_.groupby(self.gb_list[1:])[name_].transform(lambda x: (x.shift(1) - x.shift(4))/x.shift(4))
            
            self.gb_df_.fillna(-1, inplace = True)

            self.gb_df_.replace([np.inf, -np.inf], -1, inplace = True)
        
        # call the generate_shift_featues function with different suffix (sum and mean)
        generate_shift_features(self, suffix = "sum")
        generate_shift_features(self, suffix = "mean")
        
        FeatureGenerator.reduce_mem_usage(self.gb_df_)
    
        return self.gb_df_
        
def generate_shift_features(df, column_name, gb_list):
    
    df['{}_shift_1'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(1))
    df['{}_shift_2'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(2))
    df['{}_shift_3'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(3))
    df['{}_shift_6'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(6))
    
    df['{}_var_1'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(1) - x.shift(2))
    df['{}_var_2'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(1) - x.shift(3))
    df['{}_var_3'.format(column_name)] = df.groupby(gb_list)[column_name].transform(lambda x: x.shift(1) - x.shift(4))
    
    df['{}_var_pct_1'.format(column_name)]\
    = df.groupby(gb_list)[column_name].transform(lambda x: (x.shift(1) - x.shift(2))/x.shift(2))
    df['{}_var_pct_2'.format(column_name)]\
    = df.groupby(gb_list)[column_name].transform(lambda x: (x.shift(1) - x.shift(3))/x.shift(3))
    df['{}_var_pct_3'.format(column_name)]\
    = df.groupby(gb_list)[column_name].transform(lambda x: (x.shift(1) - x.shift(4))/x.shift(4))
    
    df.fillna(-1, inplace = True)

    df.replace([np.inf, -np.inf], -1, inplace = True)

    reduce_mem_usage(df)
def generate_shift_features_no_grouper(df, column_name):
    
    df['{}_shift_1'.format(column_name)] = df[column_name].shift(1)
    df['{}_shift_2'.format(column_name)] = df[column_name].shift(2)
    df['{}_shift_3'.format(column_name)] = df[column_name].shift(3)
    df['{}_shift_6'.format(column_name)] = df[column_name].shift(6)
    
    df['{}_var_1'.format(column_name)] = df[column_name].shift(1) - df[column_name].shift(2)
    df['{}_var_2'.format(column_name)] = df[column_name].shift(1) - df[column_name].shift(3)
    df['{}_var_3'.format(column_name)] = df[column_name].shift(1) - df[column_name].shift(4)
    
    df['{}_var_pct_1'.format(column_name)]\
    = (df[column_name].shift(1) - df[column_name].shift(2))/df[column_name].shift(2)
    df['{}_var_pct_2'.format(column_name)]\
    = (df[column_name].shift(1) - df[column_name].shift(3))/df[column_name].shift(3)
    df['{}_var_pct_3'.format(column_name)]\
    = (df[column_name].shift(1) - df[column_name].shift(4))/df[column_name].shift(4)
    
    df.fillna(-1, inplace = True)

    df.replace([np.inf, -np.inf], -1, inplace = True)

    reduce_mem_usage(df)
full_df.head()
full_df[full_df['date']=='2013-01-31']['item_cnt_day'].sum()
st = time.time()

def my_agg(x):
    names = {
        'month_mean': x['item_cnt_day'].mean(),
        'month_sum':  x['item_cnt_day'].sum()
    }

    return pd.Series(names, index=[key for key in names.keys()])

month_sales_features = full_df.groupby(["date"]).apply(my_agg).reset_index()

# column_name = "month_sum"
# generate_shift_features_no_grouper(month_sales_features, column_name = column_name)

# column_name = "month_mean"
# generate_shift_features_no_grouper(month_sales_features, column_name = column_name)

# et = time.time()

# (et - st)/60
month_sales_features.head()
column_name = "month_sum"
generate_shift_features_no_grouper(month_sales_features, column_name = column_name)
column_name = "month_mean"
generate_shift_features_no_grouper(month_sales_features, column_name = column_name)
month_sales_features.T
month_sales_features
month_sales_features.shape
month_sales_features.head(10).T
st = time.time()

gb_list = ["date", "shop_id"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

shop_sales_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
shop_sales_features.shape
fe_generator.objective_column_name
full_df[(full_df['shop_id']==8) & (full_df['date']=='2013-01-31')]['item_cnt_day'].sum()
shop_sales_features[shop_sales_features["shop_id"] == 8].T
st = time.time()

gb_list = ["date", "item_id"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

item_sales_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
item_sales_features.shape
item_sales_features[item_sales_features["item_id"] == 30].T
st = time.time()

gb_list = ["date", "item_category_id"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_item_category_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_item_category_features.shape
full_df[(full_df['item_category_id']==2) & (full_df[full_df['date']=='2013-01-31'])]
month_item_category_features[month_item_category_features["item_category_id"] == 2].T
st = time.time()

gb_list = ["date", "type_code"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_type_code_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_type_code_features.shape
month_type_code_features[month_type_code_features["type_code"] == 1].T
st = time.time()

gb_list = ["date", "subtype_code"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_subtype_code_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_subtype_code_features.shape
month_subtype_code_features[month_subtype_code_features["subtype_code"] == 1].T
st = time.time()

gb_list = ["date", "shop_id","type_code"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_shop_type_code_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_shop_type_code_features.shape
full_df[(full_df['shop_id']==8) & (full_df['type_code']==1) & (full_df['date']=='2013-01-31')]['item_cnt_day'].sum()
filter_1 = month_shop_type_code_features["shop_id"] == 8
filter_2 = month_shop_type_code_features["type_code"] == 1
month_shop_type_code_features[filter_1 & filter_2].T
st = time.time()

gb_list = ["date", "shop_id","subtype_code"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_shop_subtype_code_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_shop_subtype_code_features.shape
filter_1 = month_shop_subtype_code_features["shop_id"] == 8
filter_2 = month_shop_subtype_code_features["subtype_code"] == 1
month_shop_subtype_code_features[filter_1 & filter_2].T
st = time.time()

gb_list = ["date","city_code"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_city_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_city_features.shape
full_df[(full_df['city_code']==4) & (full_df['date']=='2013-01-31')]['item_cnt_day'].sum()
month_city_features[month_city_features["city_code"] == 4].T
st = time.time()

gb_list = ["date","city_code","item_id"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_city_item_id_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
month_city_item_id_features.shape
filter_1 = month_city_item_id_features["city_code"] == 4
filter_2 = month_city_item_id_features["item_id"] == 30
month_city_item_id_features[filter_1 & filter_2].T
full_df.head()
full_df.iloc[0:5,7:]
st = time.time()

gb_list = ["shop_id", "item_id"]

column_name = "item_cnt_day"
generate_shift_features(full_df, column_name = column_name, gb_list = gb_list)

et = time.time()

(et - st)/60
st = time.time()

gb_list = ["shop_id", "item_id"]

column_name = "item_price"
generate_shift_features(full_df, column_name = column_name, gb_list = gb_list)

et = time.time()

(et - st)/60
full_df.head()
st = time.time()

gb_list = ["shop_id", "item_id"]

column_name = "revenue"
generate_shift_features(full_df, column_name = column_name, gb_list = gb_list)

et = time.time()

(et - st)/60
full_df["year"] = full_df["date"].dt.year
full_df["month"] = full_df["date"].dt.month
full_df["days_in_month"] = full_df["date"].dt.days_in_month
full_df["quarter_start"] = full_df["date"].dt.is_quarter_start
full_df["quarter_end"] = full_df["date"].dt.is_quarter_end
full_df.head().T
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
full_df.head().T
full_df.rename(columns = {"item_cnt_day":"sales"}, inplace = True)
full_df.shape
full_df = pd.merge(full_df, shop_sales_features, on = ["date", "shop_id"], how = "left")


full_df = pd.merge(full_df, item_sales_features, on = ["date", "item_id"], how = "left")


full_df = pd.merge(full_df, month_sales_features, on = ["date"], how = "left")


full_df = pd.merge(full_df, month_item_category_features, on = ["date", "item_category_id"], how = "left")


full_df = pd.merge(full_df, month_type_code_features, on = ["date", "type_code"], how = "left")


full_df = pd.merge(full_df, month_subtype_code_features, on = ["date", "subtype_code"], how = "left")


full_df = pd.merge(full_df, month_shop_type_code_features, on = ["date", "shop_id", "type_code"], how = "left")


full_df = pd.merge(full_df, month_shop_subtype_code_features, on = ["date", "shop_id", "subtype_code"], how = "left")


full_df = pd.merge(full_df, month_city_features, on = ["date", "city_code"], how = "left")


full_df = pd.merge(full_df, month_city_item_id_features, on = ["date", "city_code", "item_id"], how = "left")

full_df.shape
full_df.head()
# delete dfs with features
del shop_sales_features, item_sales_features, month_sales_features, month_item_category_features,\
month_type_code_features, month_subtype_code_features, month_shop_type_code_features,\
month_shop_subtype_code_features, month_city_features, month_city_item_id_features
# delete all the previous df
del shops_df, items_df, items_category_df, sales_df, test_df, sub, cartesian_product, gb_df
gc.collect()
# st = time.time()

# full_df.to_pickle(os.path.join(PATH, "FULL_DF_ONLY_TEST_ALL_FEATURES.pkl"))

# et = time.time()

# (et - st)/60
FULL_DF_PATH = "../input/full-df-only-test-all-features/"
full_df = pd.read_pickle(os.path.join(FULL_DF_PATH, "FULL_DF_ONLY_TEST_ALL_FEATURES.pkl"))
full_df = full_df[full_df["shop_id"].isin(SHOPS)]
# delete all the columns where lags features are - 1 (shift(6))
full_df = full_df[full_df["date"] > np.datetime64("2013-06-30")]
cols_to_drop = [

'revenue',
'shop_name',
'city',
'item_name',
'item_category_name',
'split',
'type',
'subtype',
    
'date_shop_id_sum',
'date_shop_id_mean',

'date_item_id_sum',
'date_item_id_mean',

'month_mean',
'month_sum',

'date_item_category_id_sum',
'date_item_category_id_mean',

'date_type_code_sum',
'date_type_code_mean',

'date_subtype_code_sum',
'date_subtype_code_mean',

'date_shop_id_type_code_sum',
'date_shop_id_type_code_mean',

'date_shop_id_subtype_code_sum',
'date_shop_id_subtype_code_mean',

'date_city_code_sum',
'date_city_code_mean',

'date_city_code_item_id_sum',
'date_city_code_item_id_mean'
    
]
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
gc.collect()
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
# model = pickle.load(open("{}_{}.dat".format(model_name, t), "rb"))
importance = model.get_booster().get_score(importance_type = "gain")

importance = {k: v for k, v in sorted(importance.items(), key = lambda item: item[1])}
fig, ax = plt.subplots(figsize=(15, 30))
plot_importance(model, importance_type = "gain", ax = ax)
plt.savefig("{}_{}_plot_importance.png".format(model_name, t))
# model = XGBRegressor(seed = 175)

# xgbr = model.fit(X_train, Y_train, eval_metric = "rmse", 
#     eval_set = [(X_train, Y_train), (X_valida, Y_valida)], 
#     verbose = True, 
#     early_stopping_rounds = 10)
Y_valida_pred = model.predict(X_valida)
metrics.r2_score(Y_valida, Y_valida_pred)
rmse_valida = sqrt(metrics.mean_squared_error(Y_valida, Y_valida_pred))
rmse_valida
Y_test_predict = model.predict(X_test)
Y_test_predict.sum()
Y_test_predict.max()
Y_test.head()
Y_test.sum()
Y_test.max()
rmse_test = sqrt(metrics.mean_squared_error(Y_test, Y_test_predict))
rmse_test

#tentar de bajarlo
perfect_rmse = sqrt(metrics.mean_squared_error(Y_test, Y_test))
perfect_rmse
# plt.figure(figsize=(15,8))
# x_ax = range(len(Y_test))
# plt.plot(x_ax, Y_test, label="original")
# plt.plot(x_ax, Y_test_predict, label="predicted")
# plt.show()
FULL_DF_PATH = "../input/full-df-only-test-all-features/"
full_df = pd.read_pickle(os.path.join(FULL_DF_PATH, "FULL_DF_ONLY_TEST_ALL_FEATURES.pkl"))
full_df = full_df[full_df["shop_id"].isin(SHOPS)]
full_df.rename(columns = {"sales":"item_cnt_day"}, inplace = True)
# before run the class FeatureGenerator

st = time.time()

gb_list = ["date", "shop_id", "item_category_id"]

fe_generator = FeatureGenerator(full_df = full_df, gb_list = gb_list)

fe_generator.generate_gb_df()

month_shop_item_category_id_features = fe_generator.return_gb_df()

et = time.time()

(et - st)/60
full_df.rename(columns = {"item_cnt_day":"sales"}, inplace = True)
full_df = pd.merge(full_df, month_shop_item_category_id_features, on = ["date", "shop_id", "item_category_id"], how = "left")
full_df = full_df[full_df["date"] > np.datetime64("2013-06-30")]
cols_to_drop = [

'revenue',
'shop_name',
'city',
'item_name',
'item_category_name',
'split',
'type',
'subtype',
    
'date_shop_id_sum',
'date_shop_id_mean',

'date_item_id_sum',
'date_item_id_mean',

'month_mean',
'month_sum',

'date_item_category_id_sum',
'date_item_category_id_mean',

'date_type_code_sum',
'date_type_code_mean',

'date_subtype_code_sum',
'date_subtype_code_mean',

'date_shop_id_type_code_sum',
'date_shop_id_type_code_mean',

'date_shop_id_subtype_code_sum',
'date_shop_id_subtype_code_mean',

'date_city_code_sum',
'date_city_code_mean',

'date_city_code_item_id_sum',
'date_city_code_item_id_mean',
    
'date_shop_id_item_category_id_sum', #new
'date_shop_id_item_category_id_mean' #new
    
]
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
gc.collect()
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
# model = pickle.load(open("{}_{}.dat".format(model_name, t), "rb"))
# model_my = pickle.load(open('./XGBRegressor_2020_06_29_18_58_11.dat', "rb"))
Y_valida_pred = model.predict(X_valida)
metrics.r2_score(Y_valida, Y_valida_pred)
rmse_valida = sqrt(metrics.mean_squared_error(Y_valida, Y_valida_pred))
rmse_valida
Y_test_predict = model.predict(X_test)
Y_test_predict.sum()
Y_test_predict.max()
Y_test.head()
Y_test.sum()
Y_test.max()
rmse_test = sqrt(metrics.mean_squared_error(Y_test, Y_test_predict))
rmse_test
importance = model.get_booster().get_score(importance_type = "gain")

importance = {k: v for k, v in sorted(importance.items(), key = lambda item: item[1])}
fig, ax = plt.subplots(figsize=(15, 30))
plot_importance(model, importance_type = "gain", ax = ax)
plt.savefig("{}_{}_plot_importance.png".format(model_name, t))
from sklearn.ensemble import GradientBoostingRegressor
st = time.time()

gb = GradientBoostingRegressor(random_state=42, criterion='mse', verbose=1, n_iter_no_change=5)

model_name = str(gb).split("(")[0]
day = str(datetime.now()).split()[0].replace("-", "_")
hour = str(datetime.now()).split()[1].replace(":", "_").split(".")[0]
t = str(day) + "_" + str(hour)

gb.fit(X_train, Y_train)

et = time.time()

print("Training took {} minutes!".format((et - st)/60))
pickle.dump(gb, open("{}_{}.dat".format(model_name, t), "wb"))
# gb = pickle.load(open("{}_{}.dat".format(model_name, t), "rb"))
Y_valida_pred = gb.predict(X_valida)
metrics.r2_score(Y_valida, Y_valida_pred)
rmse_valida = sqrt(metrics.mean_squared_error(Y_valida, Y_valida_pred))
rmse_valida
Y_test_predict = gb.predict(X_test)
Y_test_predict.sum()
Y_test_predict.max()
Y_test.sum()
Y_test.max()
rmse_test = sqrt(metrics.mean_squared_error(Y_test, Y_test_predict))
rmse_test

