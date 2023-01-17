import sys
# import lightgbm as lgb
from  datetime import datetime, timedelta

    
import lightgbm as lgb
import os, sys, gc, time, warnings, pickle, random
from math import ceil

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth',100)
INPUT_PATH = '../../'
MAIN_INDEX = ['id','d']  # We can identify item by these columns
eval_end_day = 1941
valid_end_day = 1913

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
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
    start_mem = df.memory_usage().sum() / 1024**2  
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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


## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    # we want the extract data from df2 to add as columns in df1(i.e. left join). For saving memory(I think), just get merge_on of df1 for merging, then concat
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

def create_train_data(train_start=1,test_start=1800,is_train=True, add_test=False):
    # data types of each columns 
    # start_day = train_start if is_train else test_start
    start_day = train_start
    numcols = [f"d_{day}" for day in range(start_day,eval_end_day+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    SALE_DTYPES = {numcol:"float32" for numcol in numcols} 
    SALE_DTYPES.update({col: "category" for col in catcols if col != "id"}) 

    # loading data
    sale_data = pd.read_csv('../../sales_train_evaluation.csv',dtype=SALE_DTYPES,usecols=catcols+numcols)

    # category types to integer type
#     for col, col_dtype in PRICE_DTYPES.items():
#         if col_dtype == "category":
#             price_data[col] = price_data[col].cat.codes.astype("int16")
#             price_data[col] -= price_data[col].min()

#     cal_data["date"] = pd.to_datetime(cal_data["date"])
#     for col, col_dtype in CAL_DTYPES.items():
#         if col_dtype == "category":
#             cal_data[col] = cal_data[col].cat.codes.astype("int16")
#             cal_data[col] -= cal_data[col].min()


#     # # add test days with nan (注意提交格式里有一部分为空)
#     if not is_train:
#         for day in range(1913+1, 1913+ 2*28 +1):
#             sale_data[f"d_{day}"] = np.nan

    # In the sales dataset, each row represents one item in one specific store. since our target is sales,
    # We can tranform horizontal representation 
    # into vertical "view" so that each row represents for sales for one day.
    # Our "index" will be 'id','item_id','dept_id','cat_id','store_id','state_id'
    # and labels are 'd_' coulmns
    grid_df = pd.melt(sale_data,
            id_vars = catcols,
            value_vars = [col for col in sale_data.columns if col.startswith("d_")],
            var_name = "d",
            value_name = "sales")
    
#     # we can add test days with nan after melt, but more tedious
#     if add_test == True:
#         END_TRAIN = 1913         # Last day in train set
#         # To be able to make predictions
#         # we need to add "test set" to our grid
#         add_grid = pd.DataFrame()
#         for i in range(1,29):
#             # construct the index columns
#             temp_df = sale_data[catcols]
#             temp_df = temp_df.drop_duplicates() # Actually, no need this since each row in original sale data indexed by catcols representing one item in one store
#             # add label column for sales
#             temp_df['d'] = 'd_'+ str(END_TRAIN+i)
#             # add sales column
#             temp_df['sales'] = np.nan
#             add_grid = pd.concat([add_grid,temp_df])
#         grid_df = pd.concat([grid_df,add_grid])
#      #Remove some temoprary DFs
#         del temp_df, add_grid
        
    # the index of concated df keep the original ones, needs to be reset
    grid_df = grid_df.reset_index(drop=True)
        
    # We will not need original sale-data
    # anymore and can remove it
    del sale_data
    
    
    # It seems that leadings zero values
    # in each train_df item row
    # are not real 0 sales but mean
    # absence for the item in the store
    # by doing inner join we can remove
    # such zeros
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
            "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
            "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    cal_data = pd.read_csv('../../calendar.csv',dtype=CAL_DTYPES)
    price_data = pd.read_csv('../../sell_prices.csv',dtype=PRICE_DTYPES)
    ## get wm_yr_wk as key for join the price table
    grid_df = grid_df.merge(cal_data[['d', 'wm_yr_wk']], on= "d", copy = False)
    grid_df = grid_df.merge(price_data[["store_id", "item_id", "wm_yr_wk"]], on = ["store_id", "item_id", "wm_yr_wk"])
    
    
    return grid_df


def create_prices_features(prices_df):
    ########################### Prices
    #################################################################################
    print('Create Prices Features')

    # We can do some basic aggregations
    prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

    # Some items are can be inflation dependent
    # and some items are very "stable"
    prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique') # how many prices for each items(in a store), reflect inflation and stable
    
    # since group by object will only leave string columns, get categorical for item_id
    # also, interestingly, groupby will do combination if any element for groupby is cagtegory type
    # https://github.com/pandas-dev/pandas/issues/17594
#     prices_df['item_id_a'] =prices_df['item_id'].cat.codes
#     prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id_a'].transform('nunique') # how many items(in a store) have same price
#     prices_df = prices_df.drop('item_id_a', axis=1)

    # I would like some "rolling" aggregations, i,e. price "momentum" (some sort of)
    # but would like months and years as "window", so the next three commands add months and years as columns
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
        "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    cal_data = pd.read_csv(INPUT_PATH+'calendar.csv',dtype=CAL_DTYPES)
    ## get month, year to join into prices_df
    calendar_prices = cal_data[['wm_yr_wk','month','year']]
    # approcimately have (the length of the original/ 7), since the calendar_df is recorded by day, now week
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
    del calendar_prices

    # Now we can add price "momentum" (some sort of)
    # Shifted by week 
    # by month mean
    # by year mean
    prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1)) # the rate with sell price last day
#     ## cannot use built-in mean which would output nan if the group has nan values
#     prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean') # the rate with sell price last month
#     prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean') # the rate with sell price last year
    prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform(lambda x: np.mean([i for i in x if not np.isnan(i)]))
    prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform(lambda x: np.mean([i for i in x if not np.isnan(i)])) # the rate with sell price last year
#     # for testing the problem of transform('mean') which gives different values by using costom mean function(not because of null value)
#     idx = (price_data['store_id'] == 'WI_3') & (price_data['item_id'] == 'FOODS_3_827') & (price_data['month'] == 6)
#     price_data[['sell_price']][idx]

    del prices_df['month'], prices_df['year']
    
    prices_df = reduce_mem_usage(prices_df)
    
    return prices_df

grid_df = create_train_data()
original_columns = list(grid_df) 
# Save original sales
# grid_df.to_pickle('grid_part_1.pkl')
grid_df.head()
# add price features

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
prices_df = pd.read_csv('../../sell_prices.csv',dtype=PRICE_DTYPES)
prices_df = create_prices_features(prices_df)
grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')

del prices_df
# there are 30490 null value for price_momentum in prices_df, because of 3049 products and 10 stores in data(3049*10 records per day)
# after joining with grid_df, there are 213430 null
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        print(col, str(num_na))
        
# So by removing such rows, we acutually remove the records in the first week, I should not remove the null values here,
# Since the sell prices of the records with null price momentum could be used for created lags.
# grid_df.dropna(inplace=True)
grid_df.info()
def make_time_features(grid_df):
    # Convert to DateTime
    grid_df['date'] = pd.to_datetime(grid_df['date'])

    # Make some features from date:  
    # 有的时间特征没有，通过datetime的方法自动生成
    grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
    grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
    grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
    grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
    grid_df['tm_q'] = grid_df['date'].dt.quarter.astype(np.int8)
    grid_df['tm_y'] = grid_df['date'].dt.year
    grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
    grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)
    
    # whether it is weekend
    grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)

    # Remove date
    del grid_df['date']
    
    
    return grid_df


# add time features: from calendar files and pandas functions
## calendar files
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
        "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
cal_data = pd.read_csv(INPUT_PATH+'calendar.csv',dtype=CAL_DTYPES)
cal_data.info()
from sklearn import preprocessing
## Merge calendar partly, other generate using pandas datetime functions
icols = ['date',
         'd',
         'event_name_1',
         'event_type_1',
         'event_name_2',
         'event_type_2',
         'snap_CA',
         'snap_TX',
         'snap_WI']
grid_df = grid_df.merge(cal_data[icols], on=['d'], how='left')


## only consider SNAP for the correct state
enc = preprocessing.OneHotEncoder()
state = enc.fit_transform(grid_df[['state_id']]).toarray()
grid_df['snap'] = np.multiply(grid_df[['snap_CA', 'snap_TX', 'snap_WI']].values,state).sum(axis=1)
grid_df = grid_df.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)


grid_df = make_time_features(grid_df)
# Save part 3
# grid_df.to_pickle('grid_part_3.pkl')
grid_df.info()
for col in ['event_type_1', 'event_type_2']:
    grid_df[col] = grid_df[col].cat.codes.astype("int16")
    grid_df[col] -= grid_df[col].min()
    

grid_df = grid_df.drop(['event_name_1', 'event_name_2'], axis=1)
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        print(col, str(num_na))
def create_lags(lag_df, LAGS_SPLIT, TARGET='sales', groupby=None):
    '''Return Dataframe for lags
    Input is dataframe with last column as TARGET and others as (composite) key
    '''
    # lag creation
    # and "append" to our grid
    LAGS = []
    for LAG in LAGS_SPLIT:
        if groupby != None:
            lag_df[TARGET+'_lag_'+str(LAG)] = lag_df.groupby(groupby)[TARGET].transform(lambda x: x.shift(LAG)).astype(np.float16)
        else:
            lag_df[TARGET+'_lag_'+str(LAG)] = lag_df[TARGET].shift(LAG).astype(np.float16)
        LAGS.append(TARGET+'_lag_'+str(LAG))
        
    return lag_df[LAGS]



def make_lag_roll(roll_df, ROLS_SPLIT, groupby=None): 
    TARGET = roll_df.columns[-1]
    cols = []
    for LAG_DAY in ROLS_SPLIT:
        shift_day = LAG_DAY[0]
        roll_wind = LAG_DAY[1]
        col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
        roll_df[col_name] = roll_df.groupby(groupby)[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
        cols.append(col_name)
    return roll_df[cols]
# LAGS of sell price for the item in each stores(30490)
LAGS_SPLIT = [7, 28]
lag_df = create_lags(grid_df[['id','d','sales']], LAGS_SPLIT, groupby=['id'])


# LAG_ROLLING_WIN_STATISTICS:just average/mean here
# [[1, 7], [1, 14], [1, 30], [1, 60], [7, 7], [7, 14], [7, 30], [7, 60], [14, 7], [14, 14], [14, 30], [14, 60]]
ROLS_SPLIT = [] 
for i in [0,7,14]:
    for j in [7,14,28]:
        ROLS_SPLIT.append([i,j])
        
roll_df = make_lag_roll(grid_df[['id','d','sales']], ROLS_SPLIT, groupby=['id'])
grid_df = pd.concat([grid_df, lag_df, roll_df], axis=1)

# Save 
# pd.concat([lag_Df, roll_df], axis=1).to_pickle('lag_df.pkl')
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        print(col, str(num_na))
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        print(col, str(num_na))
# grid_df = pd.concat([pd.read_pickle('grid_part_1.pkl'),
#                      pd.read_pickle('grid_part_2.pkl').iloc[:,2:],
#                      pd.read_pickle('grid_part_3.pkl').iloc[:,2:]],
#                      axis=1)

# grid_df = pd.concat([grid_df,
#                      pd.read_pickle('lag_df.pkl')],
#                      axis=1)

# normalize category variables so that it starts from 0
# BUT we cannot do it at begining in case errors(type error, values of merge keys) when merge with price_data
# BUT I found category type save more memnory than int16
# catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
# for col in catcols:
#     if col != "id":
#         grid_df[col] = grid_df[col].cat.codes.astype("int16")
#         grid_df[col] -= grid_df[col].min()
        
# change day type to int so that it is easily manipulated
grid_df['d'] = [int(day[2:]) for day in grid_df['d']]
grid_df['d'] = grid_df['d'].astype(np.int16)
grid_df.d.unique()
grid_df.info()
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        
        print(col, str(num_na))
grid_df.to_pickle('grid_df_evaluation.pkl')
# del grid_df
