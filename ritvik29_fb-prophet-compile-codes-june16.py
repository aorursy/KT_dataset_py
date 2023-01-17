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
import os
import gc
import time
import math
import datetime

import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import scipy
import statsmodels
from fbprophet import Prophet

from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore")
start = time.time()
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
INPUT_DIR = '../input/m5-forecasting-accuracy'
calendar_data = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
price_data = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')
submission_data = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sales_data = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
evaluation_data = pd.read_csv(f'{INPUT_DIR}/sales_train_evaluation.csv')

sales_data = reduce_mem_usage(sales_data)
price_data = reduce_mem_usage(price_data)
submission_data = reduce_mem_usage(submission_data)
calendar_data = reduce_mem_usage(calendar_data)

d_cols = [c for c in sales_data.columns if 'd_' in c]


def split_train_val():
    start_day = min(calendar_data['date'])
    start_day = datetime.datetime.strptime('2011-01-29','%Y-%m-%d').date()
    dates = [start_day + i*datetime.timedelta(days =1) for i in range(0,len(d_cols))]
    train_start = 0
    train_end = dates[-29]
    return train_end




def create_snapshot(calendar_data,sales_data,id_lst):
    #from datetime import strptime
    start_day = datetime.datetime.strptime(min(calendar_data['date']),'%Y-%m-%d').date()
    dates = [start_day + i*datetime.timedelta(days =1) for i in range(0,len(d_cols))]
    gs_temp = pd.DataFrame()
    gs_snapshot = pd.DataFrame()
    for item in id_lst:
        gs_temp['snapshot_date'] = dates
        gs_temp['id'] = item
        gs_temp['cat_id']  = sales_data.loc[sales_data['id']==item]['cat_id'].values[0]
        gs_temp['store_id']  = sales_data.loc[sales_data['id']==item]['store_id'].values[0]
        gs_temp['state_id'] = sales_data.loc[sales_data['id']==item]['state_id'].values[0]
        gs_temp['item_id'] = sales_data.loc[sales_data['id']==item]['item_id'].values[0]
        gs_temp['sales'] = sales_data.loc[sales_data['id']==item][d_cols].T.values
        gs_snapshot = pd.concat([gs_snapshot,gs_temp])
    return gs_snapshot

def convert_to_wm_yr_wk(df):
    df['wm_yr_wk'] = df['snapshot_date'].apply(lambda x:str(x.isocalendar()[0])[-2:]+ 
                                               (str(x.isocalendar()[1]) if x.isocalendar()[1]>10 
                                         else '0'+ str(x.isocalendar()[1]))+str(x.isocalendar()[1]))
    return df

cpu_count()-1
def merge_price_cal_data(price_data,calendar_data):
    price_cal = price_data.merge(calendar_data, on = ['wm_yr_wk'], how = 'left')
    price_cal.rename({'date':'snapshot_date'},
                 inplace = True, axis =1)
    price_cal.drop(['wm_yr_wk','weekday','month','year'],axis = 1, inplace = True)
    price_cal['snapshot_date'] = price_cal['snapshot_date'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d').date())

    #price_cal.head()
    return price_cal

def merge_snapshot_pricecal(price_cal, gs_snapshot):
    gs_comb = gs_snapshot.merge(price_cal, on =['store_id','item_id','snapshot_date'],how = 'left')
    return gs_comb

def extract_id_info(id1):
    id_info= id1.split('_')
    state = id_info[3]
    category = id_info[0]
    return state,category


def select_snaps(gs_comb,id1):
    state, category = extract_id_info(id1)
    snap_days_CA = gs_comb[gs_comb['snap_CA']==1]['snapshot_date'].unique()
    snap_days_TX = gs_comb[gs_comb['snap_TX']==1]['snapshot_date'].unique()
    snap_days_WI = gs_comb[gs_comb['snap_TX']==1]['snapshot_date'].unique()
    if state =='CA':
        return snap_days_CA
    elif state == 'TX':
        return snap_days_TX
    else:
        return snap_days_WI


def get_train_val_dates(split,gs_comb):
    gs_comb['snapshot_date'] = gs_comb['snapshot_date'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d').date())
    start_day = datetime.datetime.strptime('2011-01-29','%Y-%m-%d').date()
    dates = [start_day + i*datetime.timedelta(days =1) for i in range(0,len(d_cols))]
    train_start = 0
    train_start = dates[0]
    train_end = dates[int(split*len(dates))]
    return train_start,train_end
    #train_end_ind = 0.9*len(dates)
    
def get_holidays(gs_comb,id1):
    Hol1_rel = gs_comb[gs_comb['event_type_1']=='Religious']['snapshot_date'].unique()
    Hol1_nat = gs_comb[gs_comb['event_type_1']=='National']['snapshot_date'].unique()
    Hol1_cul = gs_comb[gs_comb['event_type_1']=='Cultural']['snapshot_date'].unique()
    Hol1_Sp = gs_comb[gs_comb['event_type_1']=='Sporting']['snapshot_date'].unique()

    #----------------------------
    Hol2_rel = gs_comb[gs_comb['event_type_2']=='Religious']['snapshot_date'].unique()
    Hol2_cul = gs_comb[gs_comb['event_type_2']=='Cultural']['snapshot_date'].unique()
    
    #train_start, train_end = get_train_val_dates(split, gs_comb)
    snap_days1 = pd.DataFrame({
      'holiday': 'snaps',
      'ds': pd.to_datetime(select_snaps(gs_comb, id1)),
      'lower_window': 0,
      'upper_window': 0,
    })

    holiday1_rel = pd.DataFrame({
      'holiday': 'holiday_religious',
      'ds': pd.to_datetime(Hol1_rel),
      'lower_window': -1,
      'upper_window': 1,
    })



    holiday1_cul = pd.DataFrame({
      'holiday': 'holiday_cultural',
      'ds': pd.to_datetime(Hol1_cul),
      'lower_window': -1,
      'upper_window': 1,
    })

    holiday1_nat = pd.DataFrame({
      'holiday': 'holiday_national',
      'ds': pd.to_datetime(Hol1_nat),
      'lower_window': -1,
      'upper_window': 1,
    })


    holiday2_cul = pd.DataFrame({
      'holiday': 'holiday_religious',
      'ds': pd.to_datetime(Hol2_cul),
      'lower_window': -1,
      'upper_window': 1,
    })


    holiday2_rel = pd.DataFrame({
      'holiday': 'holiday_religious',
      'ds': pd.to_datetime(Hol2_rel),
      'lower_window': -1,
      'upper_window': 1,
    })
    holidays = pd.concat((snap_days1,holiday1_rel,holiday1_cul,holiday1_nat,holiday2_cul,holiday2_rel ))
    return holidays

def train_model(gs_comb,holidays, id1, train_end):
    data = gs_comb[gs_comb['id']==id1]
    data2 = data.rename({'snapshot_date':'ds','sales':'y'},axis=1)[['sell_price','ds','y']]
    data2_tr = data2[data2['ds']<=train_end]
    median =  data2_tr['sell_price'].median(axis = 0)
    data2_tr['sell_price'] = data2_tr['sell_price'].fillna(median)
    data2_tr['ds'] = data2_tr['ds'].astype('datetime64')
    m2 = Prophet(holidays=holidays,weekly_seasonality = True, yearly_seasonality= True,changepoint_prior_scale = 0.7,uncertainty_samples = True)
    m2.add_seasonality(name='monthoy', period=30.5, fourier_order=5)
    m2.add_regressor('sell_price')
    m2.fit(data2_tr)
    return m2

    
def make_prediction(m2,gs_comb,id1,train_end):
      data = gs_comb[gs_comb['id']==id1]
      data2 = data.rename({'snapshot_date':'ds','sales':'y'},axis=1)[['sell_price','ds','y']]
      data2_tr = data2[data2['ds']<=train_end]
      data2_val = data2[data2['ds']>train_end]
      n_days_val = data2_val.shape[0]
      future = m2.make_future_dataframe(periods = n_days_val)
      future['sell_price'] = np.array(data['sell_price'])
      median = data[data['snapshot_date']>train_end]['sell_price'].median(axis =0)
      future['sell_price'] = future['sell_price'].fillna(median)
      forecast2 = m2.predict(future)
      return forecast2
def make_validation_file(forecast2,id1):
    item_id = id1
    F_cols = np.array(['F'+str(i) for i in range(1,29)])
    submission = pd.DataFrame(columns=F_cols)
    submission.insert(0,'id',item_id)
    forecast2['yhat'] = np.where(forecast2['yhat']<0,0,forecast2['yhat'])
    forecast2.rename({'yhat':'y','ds':'ds'},inplace=True,axis = 1)
    forecast2_T = forecast2[['ds','y']].T
    submission.loc[1,'id'] =item_id
    submission[F_cols] = forecast2_T.loc['y',:].values[-28:]
    submission.head()
    col_order = np.insert(F_cols,0,'id')
    sub_val = submission[col_order]
    #sub_val.to_csv('submission.csv',index = False)
    return submission
#get_holidays(gs_comb,5)
#with Pool(cpu_count()) as p:
        #forecast1 = p.map(run_prophet, [temp_series])
def main1():
    id_lst = sales_data['id'].unique().tolist()
    train_end = split_train_val() 
    gs_snapshot = create_snapshot(calendar_data, sales_data)
    price_cal = merge_price_cal_data(price_data,calendar_data)
    gs_comb = merge_snapshot_pricecal(price_cal,gs_snapshot)
    state,category = extract_id_info(id_lst[5])
    print(state)
    #a1 = get_snap_days(gs_comb,5)
    hols = get_holidays(gs_comb,id_lst[5])
    model = train_model(gs_comb,hols,id_lst[5],train_end)
    forecast2 = make_prediction(model,gs_comb, id_lst[5],train_end)
    submission = make_validation_file(forecast2,id_lst[5])
    return submission
def run_prophet(gs_comb,id1):
    state,category = extract_id_info(id1)
    hols = get_holidays(gs_comb,id1)
    model = train_model(gs_comb,hols,id1,train_end)
    forecast2 = make_prediction(model,gs_comb, id1,train_end)
    submission = make_validation_file(forecast2,id1)
    return submission
def main():
    id_lst = sales_data['id'].unique().tolist()
    train_end = split_train_val() 
    gs_snapshot = create_snapshot(calendar_data, sales_data,id_lst)
    price_cal = merge_price_cal_data(price_data,calendar_data)
    gs_comb = merge_snapshot_pricecal(price_cal,gs_snapshot)
    #with Pool(4) as p:
    #   submission = p.map(run_prophet(gs_comb), id_lst[0:10])
    submission = Parallel(n_jobs=cpu_count(), verbose=3, backend="multiprocessing")(
             map(delayed(run_prophet(gs_comb)), id_lst[0:10]))
    submission2 = pd.concat(submission,axis = 0)
    submission2.to_csv('submission.csv',index = False)
    end = time.time()
    elapsed_time = end-start
    time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print('time',time_taken)
    return submission
main()