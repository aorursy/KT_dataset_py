# Models

# --------

# FBProphet

# Loader/fast aggregrations from: https://www.kaggle.com/christoffer/pandas-multi-indices-for-hts-fast-loading-etc



# FPBTD_v1 Level 9, LB=0.603

# yearly_seasonality=2 weekly_seasonality=1, n_changepoints = 50, LRMSE=12.128



# FPBTD_v2 - Levels 1 to 9

# yearly_seasonality=10 weekly_seasonality=3, n_changepoints = 50, US holidays added, LRMSE=11.824

# lag365, lag28_roll28 min and max



# FPBTD_v3 - Levels 1 to 5

# Same as previous with quarter_fourier_order=4, L6-L6=0.614



# FPBTD_v4 - Levels 1 to 9

# quarter_fourier_order=8 + cumulated price, L6-LB=0.616



# FPBTD_v5 - Levels 1 to 9

# quarter_fourier_order=6, outliers removed, L6-LB=0.637



# FPBTD_v6 - Levels 1 to 9

# quarter_fourier_order=6, outliers removed + christmas, L6-LB=0.642



# FPBTD_v7 - Levels 1 to 9

# 2013, quarter_fourier_order=6, outliers removed + christmas, L6-LB=0.61x



# FPBTD_v8 - Levels 1 to 9

# 2014, quarter_fourier_order=6, outliers removed + christmas, L6-LB=0.63x



# FPBTD_v9 - Levels 1 to 9

# 2012, quarter_fourier_order=6, outliers removed + christmas, L6-LB=0.618



# FPBTD_v10 - Levels 1 to 9

# 2012, quarter_fourier_order=6, additional lags, L6-LB=
import os, sys, random, gc, math, glob, time

import numpy as np

import pandas as pd

import io, timeit, os, gc, pickle, psutil

import joblib

from matplotlib import cm

from datetime import datetime, timedelta

import warnings

from tqdm.notebook import tqdm

from multiprocessing import Pool, cpu_count

from joblib import Parallel, delayed

from functools import partial

from collections import OrderedDict

# from tqdm.contrib.concurrent import process_map



# warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 4000)
seed = 2020

random.seed(seed)

np.random.seed(seed)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

DEFAULT_FIG_WIDTH = 20

sns.set_context("paper", font_scale=1.2) 
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, KFold, GroupKFold, ShuffleSplit

from sklearn.metrics import accuracy_score, precision_score, cohen_kappa_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

import csv

from collections import defaultdict



#import lightgbm as lgb



print('Python    : ' + sys.version.split('\n')[0])

print('Numpy     : ' + np.__version__)

print('Pandas    : ' + pd.__version__)

#print('LightGBM  : ' + lgb.__version__)
# !pip install fbprophet --upgrade
import fbprophet

from fbprophet import Prophet

print('Prophet  : ' + fbprophet.__version__)
HOME =  "./"

DATA_HOME = "/kaggle/input/m5-forecasting-accuracy/"

TRAIN_DATA_HOME = DATA_HOME



CALENDAR = DATA_HOME + "calendar.csv"

SALES = DATA_HOME + "sales_train_validation.csv"

PRICES = DATA_HOME + "sell_prices.csv"



MODELS_DIR = "models"

if not os.path.exists(MODELS_DIR):

    os.makedirs(MODELS_DIR)

            

NUM_SERIES = 30490

NUM_TRAINING = 1913

NUM_TEST = NUM_TRAINING + 2 * 28

MAX_LEVEL = 9 #9
# Load data

series_ids = np.empty(NUM_SERIES, dtype=object)

item_ids = np.empty(NUM_SERIES, dtype=object)

dept_ids = np.empty(NUM_SERIES, dtype=object)

cat_ids = np.empty(NUM_SERIES, dtype=object)

store_ids = np.empty(NUM_SERIES, dtype=object)

state_ids = np.empty(NUM_SERIES, dtype=object)



qties = np.zeros((NUM_TRAINING, NUM_SERIES), dtype=float)

sell_prices = np.zeros((NUM_TEST, NUM_SERIES), dtype=float)
# Sales

id_idx = {}

with open(SALES, "r", newline='') as f:

    is_header = True

    i = 0

    for row in csv.reader(f):

        if is_header:

            is_header = False

            continue

        series_id, item_id, dept_id, cat_id, store_id, state_id = row[0:6]

        # Remove '_validation/_evaluation' at end by regenerating series_id

        series_id = f"{item_id}_{store_id}"



        qty = np.array(row[6:], dtype=float)



        series_ids[i] = series_id



        item_ids[i] = item_id

        dept_ids[i] = dept_id

        cat_ids[i] = cat_id

        store_ids[i] = store_id

        state_ids[i] = state_id



        qties[:, i] = qty



        id_idx[series_id] = i



        i += 1
# Calendar

wm_yr_wk_idx = defaultdict(list)  # map wmyrwk to d:s

with open(CALENDAR, "r", newline='') as f:

    for row in csv.DictReader(f):

        d = int(row['d'][2:])

        wm_yr_wk_idx[row['wm_yr_wk']].append(d)
# Price

with open(PRICES, "r", newline='') as f:

    is_header = True

    for row in csv.reader(f):

        if is_header:

            is_header = False

            continue

        store_id, item_id, wm_yr_wk, sell_price = row

        series_id = f"{item_id}_{store_id}"

        series_idx = id_idx[series_id]

        for d in wm_yr_wk_idx[wm_yr_wk]:

            sell_prices[d - 1, series_idx] = float(sell_price)
# Aggregations - Levels

qty_ts = pd.DataFrame(qties,

                      index=range(1, NUM_TRAINING + 1),

                      columns=[state_ids, store_ids,

                               cat_ids, dept_ids, item_ids])



qty_ts.index.names = ['d']

qty_ts.columns.names = ['state_id', 'store_id',

                        'cat_id', 'dept_id', 'item_id']



price_ts = pd.DataFrame(sell_prices,

                        index=range(1, NUM_TEST + 1),

                        columns=[state_ids, store_ids,

                                 cat_ids, dept_ids, item_ids])

price_ts.index.names = ['d']

price_ts.columns.names = ['state_id', 'store_id',

                          'cat_id', 'dept_id', 'item_id']
LEVELS = {

    1: [],

    2: ['state_id'],

    3: ['store_id'],

    4: ['cat_id'],

    5: ['dept_id'],

    6: ['state_id', 'cat_id'],

    7: ['state_id', 'dept_id'],

    8: ['store_id', 'cat_id'],

    9: ['store_id', 'dept_id'],

    10: ['item_id'],

    11: ['state_id', 'item_id'],

    12: ['item_id', 'store_id']

}



COARSER = {

    'state_id': [],

    'store_id': ['state_id'],

    'cat_id': [],

    'dept_id': ['cat_id'],

    'item_id': ['cat_id', 'dept_id']

}
def aggregate_all_levels(df):

    levels = []

    for i in range(1, max(LEVELS.keys()) + 1):

        level = aggregate_groupings(df, i, *LEVELS[i])

        levels.append(level)

    return pd.concat(levels, axis=1)



def aggregate_groupings(df, level_id, grouping_a=None, grouping_b=None):

    """Aggregate time series by summing over optional levels



    New columns are named according to the m5 competition.



    :param df: Time series as columns

    :param level_id: Numeric ID of level

    :param grouping_a: Grouping to aggregate over, if any

    :param grouping_b: Additional grouping to aggregate over, if any

    :return: Aggregated DataFrame with columns as series id:s

    """

    if grouping_a is None and grouping_b is None:

        new_df = df.sum(axis=1).to_frame()

    elif grouping_b is None:

        new_df = df.groupby(COARSER[grouping_a] + [grouping_a], axis=1).sum()

    else:

        assert grouping_a is not None

        new_df = df.groupby(COARSER[grouping_a] + COARSER[grouping_b] +

                            [grouping_a, grouping_b], axis=1).sum()



    new_df.columns = _restore_columns(df.columns, new_df.columns, level_id,

                                      grouping_a, grouping_b)

    return new_df
def _restore_columns(original_index, new_index, level_id, grouping_a, grouping_b):

    original_df = original_index.to_frame()

    new_df = new_index.to_frame()

    for column in original_df.columns:

        if column not in new_df.columns:

            new_df[column] = None



    # Set up `level` column

    new_df['level'] = level_id



    # Set up `id` column

    if grouping_a is None and grouping_b is None:

        new_df['id'] = 'Total_X'

    elif grouping_b is None:

        new_df['id'] = new_df[grouping_a] + '_X'

    else:

        assert grouping_a is not None

        new_df['id'] = new_df[grouping_a] + '_' + new_df[grouping_b]



    new_index = pd.MultiIndex.from_frame(new_df)

    # Remove "unnamed" level if no grouping

    if grouping_a is None and grouping_b is None:

        new_index = new_index.droplevel(0)

    new_levels = ['level'] + original_index.names + ['id']

    return new_index.reorder_levels(new_levels)
agg_pd = aggregate_all_levels(qty_ts)

agg_pd.head()
# All levels

df_train = agg_pd.T.reset_index()

df_train = df_train.set_index("id") # id as index

rename_dict = {}

for c in df_train.columns:

    if c not in ['level', 'state_id', 'store_id',   'cat_id',  'dept_id',  'item_id']:

        rename_dict[c] = "d_%s" % c

df_train.rename(columns=rename_dict, inplace=True)

day_cols = pd.Series([c for c in df_train.columns if c not in ['level', 'state_id', 'store_id',   'cat_id',  'dept_id',  'item_id']]) # d_1 to d_1913

print(df_train.shape)

df_train.head()
# agg_price_pd = aggregate_all_levels(price_ts)

# agg_price_pd.head()
# df_train_price = agg_price_pd.T.reset_index()

# df_train_price = df_train_price.set_index("id") # id as index

# rename_dict = {}

# for c in df_train_price.columns:

#     if c not in ['level', 'state_id', 'store_id',   'cat_id',  'dept_id',  'item_id']:

#         rename_dict[c] = "d_%s" % c

# df_train_price.rename(columns=rename_dict, inplace=True)

# day_prices_cols = pd.Series([c for c in df_train_price.columns if c not in ['level', 'state_id', 'store_id',   'cat_id',  'dept_id',  'item_id']]) # d_1 to d_1913+

# print(df_train_price.shape)

# df_train_price.head()
# Level 12 only

df_sale = df_train[df_train["level"] == 12]

df_sale.shape
# Levels <= 9

df_train = df_train[df_train["level"] <= MAX_LEVEL]

print(df_train.shape)
# df_train_price = df_train_price[df_train_price["level"] <= MAX_LEVEL]

# print(df_train_price.shape)
# Prepare calendar columns

df_calendar = pd.read_csv(CALENDAR)

df_calendar.index = df_calendar['d'].values # d_xxx as index

df_calendar['ds'] = pd.to_datetime(df_calendar['date']) # move date as datetime in "ds" column

df_calendar['quarter'] = df_calendar['ds'].dt.quarter # add quarter feature

df_calendar.head()
# Generate holidays ds

events1 = pd.Series(df_calendar['event_name_1'].values, index=df_calendar['ds'].values).dropna()

events2 = pd.Series(df_calendar['event_name_2'].values, index=df_calendar['ds'].values).dropna()

holidays = pd.DataFrame(pd.concat([events1, events2], axis=0))

holidays['ds'] = holidays.index.values

holidays.rename({0: 'holiday'}, axis=1, inplace=True)

holidays.reset_index(drop=True, inplace=True)

del events1, events2

holidays.head()
# Clean data: remove leading zeros and outliers

def clean_data(df_train, day_cols, indx):

    t = df_train.loc[indx].copy()

    t.loc[day_cols[((t.loc[day_cols]>0).cumsum()==0).values]] = np.nan

    q1 = t.loc[day_cols].quantile(0.25)

    q3 = t.loc[day_cols].quantile(0.75)

    iqr = q3-q1

    qm = (q3+1.5*iqr)

    t.loc[day_cols][t.loc[day_cols]>qm] = qm

    return t
future_preds = 28

day_fit_cols = day_cols
# Remove noticeable dates that are not in evaluation

ignore_dates = ["2011-12-25", "2012-12-25", "2013-12-25", "2014-12-25", "2015-12-25",

                "2011-12-24", "2012-12-24", "2013-12-24", "2014-12-24", "2015-12-24",

                "2011-12-31", "2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31",

                "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01"]

ignore_days = [] # df_calendar[df_calendar['ds'].isin(ignore_dates)]["d"].apply(lambda x: int(x[2:])).values
FIRST = 338 # 1069 # 704 # 1 # 704

day_fit_cols = ["d_%d"%c for c in range(FIRST, 1914) if c not in ignore_days]

df_train = df_train[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'level'] + day_fit_cols]

df_calendar = df_calendar[df_calendar["d"].isin(["d_%d"%c for c in range(FIRST, 1942) if c not in ignore_days])]

#df_prices = df_prices[["d_%d"%c for c in range(FIRST, 1942) if c not in ignore_days]]
# df_train: "id" ["d_1", "d_2", ...]

# holidays: ["holiday", "ds"]

# df_calendar d ["date", "ds", "weekday", "wday", "month", "year", "quarter" ...]

# df_prices: "id" ["d_1", "d_2", ...]

def make_prediction(indx, model_columns = 'yhat', ret_columns = 'yhat', full_predict = True):

    global df_train, holidays, df_calendar # df_train_price, df_prices

    # full_predict = True

    # Return either series or dataframe

    # model_columns = 'yhat' # ["yhat", "yhat_lower", "yhat_upper"]  # 'yhat' ["yhat"]

    # ret_columns = model_columns # + ["ds", "y"]

    changepoints=list()

    uncertainty_samples=False # False (True to include yhat_upper ...)

    changepoint_prior_scale=0.1

    changepoint_range=0.9

    n_changepoints=50

    holidays_prior_scale=10

    yearly_seasonality=10 #2

    weekly_seasonality=3 #1

    daily_seasonality=False

    monthly_fourier_order=8

    quarter_fourier_order=6 # 6 #None

    seasonality_prior_scale=10

    seasonality_mode = 'multiplicative'  # 'additive'

    

    target = df_train.loc[indx, day_fit_cols] # sales for one time series

    # target_price = df_train_price.loc[indx, day_prices_cols] # day_fit_cols



    snap_state_id = str(df_train.loc[indx, 'state_id'])

    cols = ['ds', 'month', 'wday', 'quarter']

    if snap_state_id in ["CA", "TX", "WI"]:

        cols = cols + ['snap_'+snap_state_id]



    # Create temporary dataframe for prediction from 2011-01-29	to 2016-05-22 (d_1941) initialized with NaN for values to predict

    # ["ds", "y", "prices", "month", "wday", "quarter", "snap_xx"] (snap matching to state related to id)

    df = df_calendar.iloc[:target.shape[0]+future_preds][cols].copy()

    df['y'] = target    

    

    # Clip outliers in aggregated time series

    #q1 = df['y'].quantile(0.25)

    #q3 = df['y'].quantile(0.75)

    #iqr = q3-q1

    #qm_up = (q3+1.5*iqr)

    #qm_dw = (q1-1.5*iqr)

    #df.loc[df["y"] > qm_up, "y"] = qm_up

    #df.loc[df["y"] < qm_dw, "y"] = qm_dw

    

    #df['ft_lag365'] = df['y'].shift(365)

    #df["ft_lag365"].fillna(method ='bfill', inplace = True)

    df['ft_lag28'] = df['y'].shift(28)

    df["ft_lag28"].fillna(method ='bfill', inplace = True)

    df['ft_lag35'] = df['y'].shift(35)

    df["ft_lag35"].fillna(method ='bfill', inplace = True)

    df['ft_lag42'] = df['y'].shift(42)

    df["ft_lag42"].fillna(method ='bfill', inplace = True)

    df['ft_lag28_roll28_std'] = df['y'].shift(28).rolling(28).std()

    df["ft_lag28_roll28_std"].fillna(method ='bfill', inplace = True)

    df['ft_lag28_roll28_max'] = df['y'].shift(28).rolling(28).max()

    df["ft_lag28_roll28_max"].fillna(method ='bfill', inplace = True)

    df['ft_lag28_roll28_min'] = df['y'].shift(28).rolling(28).min()

    df["ft_lag28_roll28_min"].fillna(method ='bfill', inplace = True)

    # Add prices

    #df['prices'] = target_price.astype(np.float32)

    #df["prices"].fillna(method ='bfill', inplace = True)

    #df['ft_prices_roll7_std'] = df['prices'].rolling(7).std()

    #df["ft_prices_roll7_std"].fillna(method ='bfill', inplace = True)    



    GROWTH = 'linear' #'logistic' #'linear'

    m = Prophet(growth=GROWTH, uncertainty_samples=uncertainty_samples, changepoint_prior_scale=changepoint_prior_scale, changepoint_range=changepoint_range,

                n_changepoints = n_changepoints,

                holidays_prior_scale=holidays_prior_scale, yearly_seasonality=yearly_seasonality,

                daily_seasonality=daily_seasonality, weekly_seasonality=weekly_seasonality,

                holidays=holidays, seasonality_mode=seasonality_mode, seasonality_prior_scale=seasonality_prior_scale)

    

    m.add_country_holidays(country_name='US')



    if not monthly_fourier_order is None:

        m.add_seasonality(name='monthly', period=365.25/12, fourier_order=monthly_fourier_order)

    if not quarter_fourier_order is None:

        m.add_seasonality(name='quarterly', period=365.25/4, fourier_order=quarter_fourier_order)



    # Add regressor for month, wday, quarter (snap_XX, prices)

    for reg in df.columns:

        if reg!='ds' and reg!='y':

            m.add_regressor(reg)



    target_first_valid_index = target.first_valid_index()



    if GROWTH == "logistic":

        df["cap"] = df["y"].max()

        df["floor"] = df["y"].min()

        

    # Fit on existing data (first_valid_index = Return index for first non-NA/null value.)

    m.fit(df.loc[target.loc[target_first_valid_index:].index])



    # Remove target

    if 'y' not in ret_columns:

        df.drop(['y'], axis=1, inplace=True)



    res = None

    if full_predict == True:

        forecast = m.predict(df.loc[target_first_valid_index:]) # For all days with valid data

        forecast["yhat"] = forecast["yhat"].astype(np.float32)

        res = forecast[model_columns]

        # Update prediction from first valid index from 2016-05-22

        res.index = df.loc[target_first_valid_index:].index.values

        res = df.merge(res, left_index=True, right_index=True, how="left")[ret_columns]



    else:

        forecast = m.predict(df.iloc[-future_preds:]) # for last 28 days (2016-04-25 to 2016-05-22)

        forecast["yhat"] = forecast["yhat"].astype(np.float32)

        res = forecast[model_columns]

        # Update prediction index from d_1914 to d_1941

        res.index = df.iloc[-future_preds:].index.values



    return (indx, res)
# Basic EDA and hyperparameters tuning

# model_columns = ['yhat']

# ret_columns = model_columns + ["ds", "y"]



train_indxs = ["Total_X"] # L1

train_indxs = train_indxs + ["CA_X", "TX_X", "WI_X"] # L2

train_indxs = train_indxs + ['CA_1_X', 'CA_2_X', 'CA_3_X','CA_4_X', 'TX_1_X', 'TX_2_X', 'TX_3_X', 'WI_1_X', 'WI_2_X', 'WI_3_X'] # L3

train_indxs = train_indxs + ['HOBBIES_X', 'FOODS_X', 'HOUSEHOLD_X'] # L4

train_indxs = train_indxs + ['HOBBIES_1_X', 'HOBBIES_2_X', 'FOODS_1_X', 'FOODS_2_X', 'FOODS_3_X', 'HOUSEHOLD_1_X', 'HOUSEHOLD_2_X'] # L5



m_score = 0

for train_ind in train_indxs:

    (_, pred) = make_prediction(train_ind, model_columns = ['yhat'], ret_columns = ['yhat', 'ds', 'y']) # 'prices'

    fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 4))

    d = pred.set_index("ds").plot(kind="line", y=["y", "yhat"], ax=ax, linestyle='-', linewidth=0.8) # 'prices'

    score = np.log1p(mean_squared_error(pred["y"][:-28], pred["yhat"][:-28]))

    m_score = m_score + (score / len(train_indxs))

    plt.title("%s LRMSE=%.3f" % (train_ind, score))

    plt.show()

    #break

print("Average LRMSE=%.3f" % m_score)
print('Predicting...', flush=True)

start_time = time.time()

#pool = Pool()



train_indxs = df_train.index #[0:10]



# Memory crash with multiple CPU on Kaggle

# res = pool.map(make_prediction, train_indxs)

# res = process_map(make_prediction, train_indxs, chunksize=1)

res = []

for train_indx in tqdm(train_indxs):

    r = make_prediction(train_indx)

    res.append(r)

    

#pool.close()

#pool.join()

end_time = time.time()

print('Exec speed=%.2f' %((end_time-start_time)/train_indxs.shape[0]))
# Convert back to initial format

tmp = pd.DataFrame()

for result in res:

    uid = result[0]

    ret = result[1].rename(uid)

    tmp = pd.concat([tmp, ret], axis=1)

fbp_pd = tmp.T

fbp_pd.index.name = "id"

fbp_pd.head(10)
fbp_pd.to_pickle("model_fit.pkl.gz", compression="gzip")
yhat_df = fbp_pd.reset_index()

print(yhat_df.shape)

yhat_df.head()
df_sale = df_sale.reset_index()

df_sale.head()
# Merge back to get levels info

yhat_df = pd.merge(yhat_df, df_train.reset_index()[['id', 'level', 'state_id', 'store_id',   'cat_id',  'dept_id',  'item_id']], on=["id"], how="left")

print(yhat_df.shape)

yhat_df.head()
print(yhat_df["level"].unique())
level_coef_dict = {

    12: ["id"], # L12 x30490

    10: ["item_id"], # L10 x3049

    5: ["dept_id"], # L5 x7

    4: ["cat_id"], # L4 x3

    3: ["store_id"], # L3 x10

    2: ["state_id"],  # L2 x3

    1: ["all"],  # L1 x1

    11: ["state_id", "item_id"], # L11 x9167

    7: ["state_id", "dept_id"],  # L7 x21

    9: ["store_id","dept_id"],  # L9 x70

    6: ["state_id", "cat_id"], # L6 x 9

    8: ["store_id","cat_id"], # L8 x30

}
# Top-Down prediction

def predict_topdown(df, df_level12, items):

    df_level_forecast = pd.DataFrame()

    for idx, row in df.iterrows():

        item1 = items[0]



        if item1 is not "all":

            item1_id = row[item1]

        if len(items) == 2:

            item2 = items[1]

            item2_id = row[items[1]]



        if item1 is not "all":

            # Find all level 12 items for the dept_id, store_id pair

            if len(items) == 2:

                df_item = df_level12.loc[(df_level12[item1] == item1_id) & (df_level12[item2] == item2_id)][['id']]

            else:

                df_item = df_level12.loc[df_level12[item1] == item1_id][['id']]

        else:

            df_item = df_level12[['id']]

        #print(df_item.shape)

        #display(df_item.head())



        # Sum sales from last 28 days in level 12 training

        if item1 is not "all":

            if len(items) == 2:

                df_item['val'] = df_level12[(df_level12[item1] == item1_id) & (df_level12[item2] == item2_id)].iloc[:, np.r_[0,-28:0]].sum(axis = 1)

            else:

                df_item['val'] = df_level12[df_level12[item1] == item1_id].iloc[:, np.r_[0,-28:0]].sum(axis = 1)

        else:

            df_item['val'] = df_level12.iloc[:, np.r_[0,-28:0]].sum(axis = 1)

        #display(df_item.head())



        # Back to per id prediction

        for i in range(1,29):

            col = "d_%d" % (1913 + i)

            p_col = "F%d" % i

            df_item[p_col] = (df_item['val'] * float(row[col]) / df_item['val'].sum())

        #display(df_item.head())



        df_level_forecast = pd.concat([df_level_forecast, df_item])

    return df_level_forecast.drop(columns=["val"])
for key, value in level_coef_dict.items():

    if (key <= MAX_LEVEL) and (key >= 1):

        predictions = yhat_df[yhat_df["level"] == key]

        df_levelx_forecast = predict_topdown(predictions, df_sale, value)

        print("Top-Down prediction for level %s, %s, items=%s into %s" % (key, value, predictions.shape[0], df_levelx_forecast.shape[0]))

        df_levelx_forecast.to_pickle(MODELS_DIR + "/level_%d.pkl.gz" % key, compression="gzip")

        # display(df_levelx_forecast.head())

print(df_levelx_forecast.shape)

df_levelx_forecast.head()
tmp_pd1 = df_levelx_forecast[df_levelx_forecast["id"].isin(["FOODS_1_001_CA_1"])]

tmp_pd1 = pd.melt(frame = tmp_pd1, 

                  id_vars = ['id'],

                  var_name = "F",

                  value_vars = [c for c in tmp_pd1.columns if "F" in c],

                  value_name = "yhat")

d = tmp_pd1.plot(kind="line", x="F", y="yhat")
SAMPLE_SUBMISSION = TRAIN_DATA_HOME + "sample_submission.csv"

sub_ids = pd.read_csv(SAMPLE_SUBMISSION)[['id']]

sub_ids = sub_ids[sub_ids.id.str.endswith("validation")]
for key, value in level_coef_dict.items():

    if (key <= MAX_LEVEL) and (key >= 1):

        predictions = yhat_df[yhat_df["level"] == key]

        df_levelx_forecast = predict_topdown(predictions, df_sale, value)

        print("Top-Down prediction for level %s, %s, items=%s into %s" % (key, value, predictions.shape[0], df_levelx_forecast.shape[0]))

        df_levelx_forecast["id"] = df_levelx_forecast["id"].astype(str) + "_validation"

        part1 = pd.merge(sub_ids, df_levelx_forecast, on="id", how="left")

        part2 = part1.copy()

        part2["id"] = part2["id"].str.replace("validation$", "evaluation")

        sub = pd.concat([part1, part2])

        MODE_LEVEL_PATH = MODELS_DIR + "/level_%d" % key

        if not os.path.exists(MODE_LEVEL_PATH):

            os.makedirs(MODE_LEVEL_PATH)

        sub.to_csv(MODE_LEVEL_PATH + "/submission.csv.gz", compression="gzip", index=False)
print(sub.shape)
sub.head()
sub.tail()