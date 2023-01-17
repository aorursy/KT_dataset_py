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
import pandas as pd

import numpy as np

from fbprophet import Prophet

from tqdm import tqdm, tnrange

from multiprocessing import Pool, cpu_count

import functools

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
calendar_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_data =  pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

price_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')



submission0 = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

sales_data.shape
#sales_data = evaluation

status = 'validation'
price_data['idx'] = range(price_data.shape[0])

price_data_max = price_data.groupby(['item_id']).agg({'sell_price':'max'})

price_data_max.rename({'sell_price':'sell_price_max'},inplace = True, axis = 1)

price_data_max.reset_index(inplace=True)

price_data = price_data.merge(price_data_max,on = ['item_id'], how = 'left')

#price_data['discount'] =100* (price_data['sell_price_max'] - price_data['sell_price'])/price_data['sell_price_max']

price_data['log_sell_price'] = np.log(price_data['sell_price'])

price_data['id']  = price_data['item_id'] + pd.Series(np.repeat('_',price_data.shape[0])) +price_data['store_id'] + pd.Series(np.repeat('_'+str(status),price_data.shape[0]))

price_data['id']  = price_data['item_id'] + pd.Series(np.repeat('_',price_data.shape[0])) +price_data['store_id'] + pd.Series(np.repeat('_'+str(status),price_data.shape[0]))

#price_data = price_data[['id','wm_yr_wk','discount']]

price_data = price_data[['id','wm_yr_wk','log_sell_price']]

price_data = price_data.merge(calendar_data[['wm_yr_wk','d']], how = 'left', on = 'wm_yr_wk')

price_data = reduce_mem_usage(price_data)

sales_data = reduce_mem_usage(sales_data)

price_data['did'] = price_data['d'].apply(lambda x:x.split('_')[1])

price_data['did'] = price_data['did'].astype('int32')







common_cols = 6

end_time = sales_data.shape[1] - common_cols

start_time = end_time -2*365

start_idx = start_time + 5

start_idx





def extract_id_info(id1):

    id_info= id1.split('_')

    state = id_info[3]

    category = id_info[0]

    return state,category





def select_snaps(df,id1):

    state, category = extract_id_info(id1)

    snap_days_CA = df[df['snap_CA']==1]['date'].unique()

    snap_days_TX = df[df['snap_TX']==1]['date'].unique()

    snap_days_WI = df[df['snap_TX']==1]['date'].unique()

    if state =='CA':

        return snap_days_CA

    elif state == 'TX':

        return snap_days_TX

    else:

        return snap_days_WI
def get_holidays(id1):



    Hol1_rel = calendar_data[calendar_data['event_type_1']=='Religious']['date'].unique()

    Hol1_nat = calendar_data[calendar_data['event_type_1']=='National']['date'].unique()

    Hol1_cul = calendar_data[calendar_data['event_type_1']=='Cultural']['date'].unique()

    Hol1_Sp = calendar_data[calendar_data['event_type_1']=='Sporting']['date'].unique()



    #----------------------------

    Hol2_rel = calendar_data[calendar_data['event_type_2']=='Religious']['date'].unique()

    Hol2_cul = calendar_data[calendar_data['event_type_2']=='Cultural']['date'].unique()    

    

    

    snap_days1 = pd.DataFrame({

      'holiday': 'snaps',

      'ds': pd.to_datetime(select_snaps(calendar_data, id1)),

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

    

    

    holidays =  pd.concat((snap_days1,holiday1_rel,holiday1_cul,holiday1_nat,holiday2_cul,holiday2_rel))

    return holidays







def run_prophet(id1,data):

    holidays = get_holidays(id1)

    model = Prophet(uncertainty_samples=False,

                    holidays=holidays,

                    weekly_seasonality = True,

                    yearly_seasonality= True,

                    changepoint_prior_scale = 0.5

                   )

    

    model.add_seasonality(name='monthly', period=30.5, fourier_order=2)

    model.add_regressor('log_sell_price')

    try:

        model.fit(data)

        future = model.make_future_dataframe(periods=28, include_history=False)

        future['log_sell_price'] = np.repeat(data['log_sell_price'].iloc[-1],28)

        forecast2 = model.predict(future)

        submission = make_validation_file(id1,forecast2)

        return submission

    

    except:

        print('Failed-**************',id1)

        COLS = submission0.columns[0:]

        dd = np.hstack([np.array(id1),np.ones(28)]).reshape(1,29)

        submission = pd.DataFrame(dd,columns = COLS)

        return submission

    

    





F_cols = np.array(['F'+str(i) for i in range(1,29)])



def make_validation_file(id1,forecast2):

    item_id = id1

    submission = pd.DataFrame(columns=F_cols)

    submission.insert(0,'id',item_id)

    forecast2['yhat'] = np.where(forecast2['yhat']<0,0,forecast2['yhat'])

    forecast2.rename({'yhat':'y','ds':'ds'},inplace=True,axis = 1)

    forecast2 = forecast2[['ds','y']].T

    submission.loc[1,'id'] =item_id

    submission[F_cols] = forecast2.loc['y',:].values[-28:]

    #col_order = np.insert(F_cols,0,'id')

    #sub_val = submission[col_order]

    return submission






def fill_missing_did(price_series):

    did_a1 = price_series['did'].unique()

    did_range = range(start_time, end_time+1)

    missing_dids = [i for i in did_range if i not in did_a1]

    len_missing = len(missing_dids)

    #mode_discount = price_series['discount'].mode()[0]

    mode_sell_price = price_series['log_sell_price'].mode()[0]

    missing_t = pd.DataFrame()



    missing_t['id'] = np.repeat(id1,len_missing)

    #missing_t['discount'] = np.repeat(0.0,len_missing)

    missing_t['log_sell_price'] = np.repeat(0.0,len_missing)

    missing_t['did'] = np.array(missing_dids)

    price_series = pd.concat([price_series,missing_t])

    price_series = price_series.sort_values(['did'],ascending = True)

    return price_series



data_m =[]

test = pd.DataFrame()

id_lst =[]

dids = list(range(start_time,end_time+1))

counter = 0

price_data = price_data[['id','log_sell_price','did']]

for i in tnrange(sales_data.shape[0]):

    id1 = sales_data.iloc[i,0]

    id_lst.append(id1)

    #print(id1)

    data_series = sales_data.iloc[i,start_idx:]

    price_series = price_data[(price_data['id']==id1)]

    #-----------------------------------------------------------------------------

    price_series = price_series[(price_series['did']>=start_time) & (price_series['did']<=end_time)]

    price_series = fill_missing_did(price_series)

    

    #price_series = price_series[['discount']]

    price_series = price_series[['log_sell_price']]

    price_series.index = calendar_data['date'][start_idx:start_idx+len(data_series)]

    data_series.index = calendar_data['date'][start_idx:start_idx+len(data_series)]

    data_series =  pd.DataFrame(data_series)

    data_series = data_series.reset_index()

    price_series = price_series.reset_index()

    price_series.rename({'date':'ds'},inplace = True, axis = 1)

    data_series.columns = ['ds', 'y']

    data_series = data_series.merge(price_series, how = 'left', on = 'ds')

    data_series = data_series[['ds','log_sell_price','y']]

    #data_series = data_series[['ds','y']]

    data_m.append(data_series)

    counter +=1

    if id1=='FOODS_3_068_WI_2_validation':

        test = data_series

    

comb_lst = [(id_lst[counter],data_m[counter]) for counter in range(0,len(id_lst))]

#run_prophet(comb_lst[-1][0],comb_lst[-1][1])

#comb_lst[-100:]
#with open('comb_lst.npy', 'wb') as f:

#    np.save(f, np.array(comb_lst))

#import numpy as np

#comb_lst = np.load('../input/comb-lst/comb_lst.npy', allow_pickle = True)

#len(comb_lst)
from joblib import Parallel, delayed

import time

start = time.time()

submission = Parallel(n_jobs=4, backend="multiprocessing")(delayed(run_prophet)(comb_lst[i][0],comb_lst[i][1]) for i in range(len(comb_lst)))

submission = pd.concat(submission,axis =0)

end = time.time()

elapsed_time = end-start

time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print('time',time_taken)
"""

import time

start = time.time()

with Pool(4) as p:

    submission = p.starmap(run_prophet,comb_lst[0:16000])



#print(id1l)

submission = pd.concat(submission,axis =0)

end = time.time()

elapsed_time = end-start

time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print('time',time_taken)

"""
submission.to_csv('submission_validation_Log_price_July9_1.csv',index = False)

submission.head
F_cols


def transform(df):

    newdf = df.melt(id_vars=["id"], var_name="d", value_name="sale")

    newdf.sort_values(by=['id', "d"], inplace=True)

    newdf.reset_index(inplace=True)

    return newdf



from sklearn.metrics import mean_squared_error



def rmse(df, gt):

    df = transform(df)

    gt = transform(gt)

    return mean_squared_error(df["sale"], gt["sale"])



def get_rmse(sales_data):

    DAYS_PRED =28

    start_date_val = 1913 -28

    start_date_eval = 1941-28

    dayCols = ["d_{}".format(i) for i in range(start_date_val, start_date_val+DAYS_PRED)]

    gt = sales_data[["id"]+dayCols]



    dayCols = ["F{}".format(i) for i in range(1, DAYS_PRED+1)]

    df = submission[["id"]+dayCols]

    print("RMSE",rmse(df, gt))
sales_data.head()