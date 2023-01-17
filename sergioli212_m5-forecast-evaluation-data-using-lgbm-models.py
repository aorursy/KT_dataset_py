import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
from math import ceil
from datetime import datetime
from  datetime import datetime, timedelta


INPUT_DIR = '/kaggle/input/m5-forecast/'
INPUT_DIR2 = '/kaggle/input/m5-forecasting-accuracy/'
features_columns = ['item_id',
 'dept_id',
 'cat_id',
 'sell_price',
 'price_max',
 'price_min',
 'price_std',
 'price_mean',
 'price_norm',
 'price_nunique',
 'price_momentum',
 'price_momentum_m',
 'price_momentum_y',
 'event_type_1',
 'event_type_2',
 'snap',
 'tm_d',
 'tm_dw',
 'tm_w',
 'tm_m',
 'tm_q',
 'tm_y',
 'tm_wm',
 'tm_w_end',
 'sales_lag_7',
 'sales_lag_28',
 'rolling_mean_tmp_0_7',
 'rolling_mean_tmp_0_14',
 'rolling_mean_tmp_0_28',
 'rolling_mean_tmp_7_7',
 'rolling_mean_tmp_7_14',
 'rolling_mean_tmp_7_28',
 'rolling_mean_tmp_14_7',
 'rolling_mean_tmp_14_14',
 'rolling_mean_tmp_14_28']

 

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

eval_end_day = 1941 # the last day in evaluation data set
def create_test_data(test_start=1800,is_train=True, add_test=False):
    # data types of each columns 
    # start_day = train_start if is_train else test_start
    numcols = [f"d_{day}" for day in range(test_start,eval_end_day+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    SALE_DTYPES = {numcol:"float32" for numcol in numcols} 
    SALE_DTYPES.update({col: "category" for col in catcols if col != "id"}) 

    # loading data
    sale_data = pd.read_csv(INPUT_DIR2 + 'sales_train_evaluation.csv',dtype=SALE_DTYPES,usecols=catcols+numcols)


    # # add test days with nan 
    for day in range(eval_end_day+1, eval_end_day+ 28 +1): # 1942 to 1942+28
        sale_data[f"d_{day}"] = np.nan

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
    cal_data = pd.read_csv(INPUT_DIR2 + './calendar.csv',dtype=CAL_DTYPES)
    price_data = pd.read_csv(INPUT_DIR2 + '/sell_prices.csv',dtype=PRICE_DTYPES)
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
    cal_data = pd.read_csv(INPUT_DIR2+'calendar.csv',dtype=CAL_DTYPES)
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
grid_df = create_test_data()

# add price features
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
prices_df = pd.read_csv(INPUT_DIR2 + '/sell_prices.csv',dtype=PRICE_DTYPES)
prices_df = create_prices_features(prices_df)
grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
del prices_df
# These 16 items in the specific stores are lack of sell_prices in one specific wm_yr_wk. So totally 16*7=112
# ['HOUSEHOLD_1_183_CA_4_evaluation', 'FOODS_3_296_CA_1_evaluation',
#        'FOODS_3_296_CA_2_evaluation', 'FOODS_3_296_TX_2_evaluation',
#        'FOODS_3_296_WI_2_evaluation', 'HOUSEHOLD_1_512_CA_3_evaluation',
#        'FOODS_3_296_CA_4_evaluation', 'HOUSEHOLD_1_400_WI_2_evaluation',
#        'FOODS_3_595_CA_1_evaluation', 'HOUSEHOLD_1_311_CA_2_evaluation',
#        'HOUSEHOLD_1_405_CA_2_evaluation', 'HOUSEHOLD_1_278_CA_3_evaluation', 'FOODS_3_595_CA_3_evaluation',
#        'HOUSEHOLD_1_400_CA_4_evaluation', 'HOUSEHOLD_1_386_WI_1_evaluation','HOUSEHOLD_1_020_WI_2_evaluation']

#  853720 = 30490 items in 10 stores * 28 days which would be predicted
 
for col in list(grid_df):
    num_na = grid_df[col].isnull().sum()
    if num_na != 0:
        print(col, str(num_na))
# check some anomolous records
# idx = (grid_df.d=='d_1812') & (grid_df.id == 'HOUSEHOLD_1_183_CA_4_evaluation')
# grid_df[idx]
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
    #del grid_df['date']
    
    
    return grid_df
# add time features: from calendar files and pandas functions
## calendar files
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
        "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
cal_data = pd.read_csv(INPUT_DIR2+'calendar.csv',dtype=CAL_DTYPES)
cal_data.info()


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
grid_df.columns
# Now, wwe only need to consider lags features which would be predicted each and then used to predict the next one
[col for col in features_columns if col not in grid_df.columns]
for col in ['event_type_1', 'event_type_2']:
    grid_df[col] = grid_df[col].cat.codes.astype("int16")
    grid_df[col] -= grid_df[col].min()
    
# change day type to int so that it is easily manipulated
grid_df['d'] = [int(day[2:]) for day in grid_df['d']]
grid_df['d'] = grid_df['d'].astype(np.int16)
for col in [col for col in features_columns if col not in grid_df.columns]:
    grid_df[col] = -1
def predict_for_store(test_data, train_cols,m_lgb, alpha):
    '''
    Input: test_data: the data in one store with 3049 items
    Outout: test-data with predicted sales
    '''
    

    date = datetime(2016,5, 23) 
    for i in range(0, 28):
        day = date + timedelta(days=i)
        print(i, day)

        # LAGS of sell price for the item in each stores(30490)
        LAGS_SPLIT = [7, 28]
        lag_cols = [f"sales_lag_{lag}" for lag in LAGS_SPLIT ]
        #lag_df = create_lags(grid_df[['id','d','sales']], LAGS_SPLIT, groupby=['id'])
        for lag, lag_col in zip(LAGS_SPLIT, lag_cols):
            test_data.loc[test_data.date == day, lag_col] = test_data.loc[test_data.date ==day-timedelta(days=lag), 'sales'].values   # 3049 items

            # cannot use test_data.shift(lag) or  test_data.groupby('item_id').transform(lambda x: x.shift(7)) 
            # since there are 3049* n days 'sales' for 3049 items
            # test_data = grid_df[grid_df.store_id == 'CA_1']
            # day = datetime(2016,5, 23) 
            # test_data[['sales','item_id']].groupby('item_id').apply(lambda x: x.shift(7))# all the days

        
        # LAG_ROLLING_WIN_STATISTICS:just average/mean here
        for win in [7,14,28]:
            for lag in [0,7,14]:
                df_window = test_data[(test_data.date <= day-timedelta(days=lag)) & (test_data.date > day-timedelta(days=lag+win))]
                df_window_grouped = df_window.groupby("item_id").agg({'sales':'mean'})
                #df_window_grouped = df_window_grouped.reindex(test_data.loc[test_data.date==day,'item_id']) # I am not sure why it appears, but someone adds this
                test_data.loc[test_data.date == day,f"rolling_mean_tmp_{lag}_{win}"] = df_window_grouped.sales.values  

        
        test = test_data.loc[test_data.date == day , train_cols]
        test_data.loc[test_data.date == day, "sales"] = alpha*m_lgb.predict(test) # predict 3049 items in that day in that store
        
#     return test_data.loc[(test_data.date >= date) & (test_data.date < date+timedelta(days=28)), 'sales']
    return test_data.loc[(test_data.date >= date), ['id', 'd', 'sales']]



alphas = [1.035, 1.03, 1.028,1.025,  1.023, 1.02,  1.025]
alpha = 1.018
# for alpha in alphas:
VER = 2
stores = ['CA_1','CA_2', 'CA_3','CA_4',  'TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']
submission_each_store = []
for store in stores:


    test_data = grid_df[grid_df.store_id == store]
    # load the model
    model_path = 'lgb_model_'+store+'_v'+str(VER)+'.bin' 
    model_path = INPUT_DIR + model_path
    model = pickle.load(open(model_path, 'rb'))
    test_data = predict_for_store(test_data, features_columns,model, alpha) # 30490 items for 1 store in 28days



    # test_data["F"] = [f"F{rank}" for rank in test_data.groupby("id")["id"].cumcount()+1]
    test_data['F'] = [f"F{rank}" for rank in test_data['d']-1941]
    # after checking there is no null value, otherwise, fillna: test_sub.fillna(0., inplace = True)
    # test_data[test_data.sales.isnull()] 
    test_data = test_data.set_index(["id", "F" ]).unstack()['sales'][ [f"F{i}" for i in range(1,29)]]

    submission_each_store.append(test_data)

submission_df_eval = pd.concat(submission_each_store)
# join with right order in samle_submission
sub = pd.read_csv(INPUT_DIR2+'sample_submission.csv')
submission_df_eval = submission_df.loc[list(sub[30490:]['id']),:].reset_index()
    
# get ground true of validation
sales_evaluation = pd.read_csv(INPUT_DIR2+'sales_train_evaluation.csv')
d_cols = [f'd_{str(day)}' for day in range(1914, 1942)]
sales_evaluation = sales_evaluation[['id'] + d_cols]
# change column names to F_#
new_d_cols = [f'F{day}'  for day in range(1,29)]
sales_evaluation.columns = ['id'] + new_d_cols
# change id name
sales_evaluation['id'] = sales_evaluation["id"].str.replace("evaluation$", "validation")

sales_evaluation.to_csv('validation_ground_truth.csv', index=False)
# join result of validation and evaluation
pd.concat([sales_evaluation, submission_df_eval]).to_csv('submission_'+str(alpha)+'.csv', index=False)




# Calculate by weight
# store = 'CA_1'
# test_data = grid_df[grid_df.store_id == store ]
# # load the model
# model_path = 'lgb_model_'+store+'_v'+str(2)+'.bin' 
# model_path = INPUT_DIR + model_path
# model = pickle.load(open(model_path, 'rb'))
# test_data = predict_for_store(test_data, features_columns,model, 1.018) # 30490 items for 1 store in 28days



# # test_data["F"] = [f"F{rank}" for rank in test_data.groupby("id")["id"].cumcount()+1]
# test_data['F'] = [f"F{rank}" for rank in test_data['d']-1941]
# # after checking there is no null value, otherwise, fillna: test_sub.fillna(0., inplace = True)
# # test_data[test_data.sales.isnull()] 
# # test_data = test_data.set_index(["id", "F" ]).unstack()['sales'][ [f"F{i}" for i in range(1,29)]].reset_index()