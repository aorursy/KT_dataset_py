# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime



import pyarrow.parquet

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Simple "Memory profilers" to see memory usage

def get_memory_usage():

    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

        

def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)
## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

def reduce_mem_usage(df, verbose=True):

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

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
sales_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

dates = pd.DataFrame(columns = ['d_'+str(datum) for datum in range(1914,1970)])

all_sales = pd.merge(sales_data,dates,left_index=True,right_index=True,how = "outer")

all_sales.fillna(0,inplace = True)

index_columns = ["id","item_id","dept_id","cat_id","store_id","state_id"]

sales_data_melted = all_sales.melt(id_vars= index_columns,

    value_vars= [col for col in all_sales.columns if col.startswith("d_")],

    var_name="d",

    value_name='sales')

del sales_data,all_sales,dates



# Let's check our memory usage

print("{:>20}: {:>8}".format('Original df',sizeof_fmt(sales_data_melted.memory_usage(index=True).sum())))



# We can free some memory 

# by converting "strings" to categorical

# it will not affect merging and 

# we will not lose any valuable data

for col in index_columns:

    sales_data_melted[col] = sales_data_melted[col].astype('category')



# Let's check again memory usage

print("{:>20}: {:>8}".format('Reduced df',sizeof_fmt(sales_data_melted.memory_usage(index=True).sum())))
sales_data_melted.tail()
sales_data_melted.groupby('id')['d'].count()
price_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

price_data.groupby(["store_id","item_id"])["wm_yr_wk"].count()
release_week = price_data.groupby(["store_id","item_id"])["wm_yr_wk"].min().reset_index()

release_week.columns = ["store_id","item_id","release_wk"]

calender_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv",usecols = ["wm_yr_wk","d"])

calender_data.drop_duplicates(inplace =True)



df_temp = pd.merge(sales_data_melted,release_week, on = ["store_id","item_id"],how = "left")

sales_data_release = pd.merge(df_temp,calender_data, on = ["d"], how = "left")



del release_week,calender_data,sales_data_melted,df_temp,price_data

sales_data_release.head()
for col in index_columns:

    sales_data_release[col] = sales_data_release[col].astype('category')
sales_data_release.info()
# Let's check our memory usage

print("{:>20}: {:>8}".format('Original df',sizeof_fmt(sales_data_release.memory_usage(index=True).sum())))



# We can free some memory 

# by converting "strings" to categorical

# it will not affect merging and 

# we will not lose any valuable data



sales_data_release = sales_data_release[sales_data_release["release_wk"] <= sales_data_release["wm_yr_wk"]]



# Let's check again memory usage

print("{:>20}: {:>8}".format('Reduced df',sizeof_fmt(sales_data_release.memory_usage(index=True).sum())))
price_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

price_data.head()

df_sales_prices = pd.merge(sales_data_release,price_data,on = ["store_id","item_id","wm_yr_wk"],how = "left")

del price_data, sales_data_release

df_sales_prices.head()
%whos DataFrame


# Let's check our memory usage

print("{:>20}: {:>8}".format('Original df',sizeof_fmt(df_sales_prices.memory_usage(index=True).sum())))

df_sales_prices = reduce_mem_usage(df_sales_prices)

# Let's check again memory usage

print("{:>20}: {:>8}".format('Reduced df',sizeof_fmt(df_sales_prices.memory_usage(index=True).sum())))
for col in index_columns:

    df_sales_prices[col] = df_sales_prices[col].astype('category')

df_sales_prices.info()
calender_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

calender_data.info()
icols = ['event_name_1',

         'event_type_1',

         'event_name_2',

         'event_type_2',

         'snap_CA',

         'snap_TX',

         'snap_WI',

         'wday',

         'month'

        ]

for col in icols:

    calender_data[col] = calender_data[col].astype('category')

calender_data["date"] = pd.to_datetime(calender_data["date"])

calender_data["day"] = calender_data["date"].dt.day.astype(np.int8)

calender_data["year"] = calender_data["year"].astype(np.int16)

calender_data.drop(["wm_yr_wk","weekday","date"],axis = 1,inplace =True) 

calender_data.info()
%whos DataFrame
df_sales_prices_calender = pd.merge(df_sales_prices,calender_data,on = "d",how = "left")

del df_sales_prices,calender_data

df_sales_prices_calender.head()
dates_training = pd.DataFrame([['d_'+str(datum),"train"] for datum in range(1,1914)],columns = ["d","set"])

dates_validation = pd.DataFrame([['d_'+str(datum),"valid"] for datum in range(1914,1942)],columns = ["d","set"])

dates_evaluation = pd.DataFrame([['d_'+str(datum),"eval"] for datum in range(1942,1970)],columns = ["d","set"])

df_sets = pd.concat([dates_training,dates_validation,dates_evaluation])

df_sets["set"] = df_sets["set"].astype('category')

del dates_training,dates_validation,dates_evaluation
df_sales_prices_calender_all = pd.merge(df_sales_prices_calender,df_sets,on = "d")

del df_sales_prices_calender,df_sets
df_sales_prices_calender_all.info()
df_sales_prices_calender_all.to_pickle("df_sales_prices_calender.pkl")
df_sales_prices_calender_all.tail()
%whos DataFrame