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
calendar_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

#sales_data =  pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

submission0 = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

#submission_m = pd.read_csv('../input/subjune25/submission.csv')

evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
#temp = submission0.iloc[submission_m.shape[0]:,:]

#temp2 = pd.concat([submission_m,temp])

#temp2.to_csv('temp2.csv')

#temp2.shape



sales_data = evaluation

sales_data.shape
sales_data.head()
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

"""

Hol1_rel = df[df['event_type_1']=='Religious']['snapshot_date'].unique()

Hol1_nat = df[df['event_type_1']=='National']['snapshot_date'].unique()

Hol1_cul = df[df['event_type_1']=='Cultural']['snapshot_date'].unique()

Hol1_Sp = df[df['event_type_1']=='Sporting']['snapshot_date'].unique()



#----------------------------

Hol2_rel = df[df['event_type_2']=='Religious']['snapshot_date'].unique()

Hol2_cul = df[df['event_type_2']=='Cultural']['snapshot_date'].unique()







"""
def run_prophet(id1,data):

    holidays = get_holidays(id1)

    model = Prophet(uncertainty_samples=False,

                    holidays=holidays,

                    weekly_seasonality = True,

                    yearly_seasonality= True,

                    changepoint_prior_scale = 0.7

                   )

    

    #model.add_seasonality(name='monthly', period=30.5, fourier_order=2)

    model.fit(data)

    future = model.make_future_dataframe(periods=28, include_history=False)

    forecast2 = model.predict(future)

    submission = make_validation_file(id1,forecast2)

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





"""

def train_model(data,holidays, id1,train_start, train_end):

    data = data[data['id']==id1]

    data = data.rename({'snapshot_date':'ds','sales':'y'},axis=1)[['sell_price','ds','y']]

    data_tr = data[(data['ds']>=str(train_start)) & (data['ds']<=str(train_end))]

    median =  data_tr['sell_price'].median(axis = 0)

    data_tr['sell_price'] = data_tr['sell_price'].fillna(median)

    data_tr['ds'] = data_tr['ds'].astype('datetime64')

    m2 = Prophet(holidays=holidays,weekly_seasonality = True, yearly_seasonality= True,changepoint_prior_scale = 0.7,uncertainty_samples = False)

    m2.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    m2.add_regressor('sell_price')

    m2.fit(data_tr)

    return m2

"""







"""

id1 = sales_data.iloc[0,0]

data_series = sales_data.iloc[1,start_idx:]

data_series.index = calendar_data['date'][start_idx:start_idx+len(data_series)]

data_series =  pd.DataFrame(data_series)

data_series = data_series.reset_index()

data_series.columns = ['ds', 'y']

data_series

"""
common_cols = 6

end_time = sales_data.shape[1] - common_cols

start_time = end_time -2*365

start_idx = start_time + 5

data_m =[]

id_lst =[]

for i in tnrange(sales_data.shape[0]):

    id_lst.append(sales_data.iloc[i,0])

    data_series = sales_data.iloc[i,start_idx:]

    data_series.index = calendar_data['date'][start_idx:start_idx+len(data_series)]

    data_series =  pd.DataFrame(data_series)

    data_series = data_series.reset_index()

    data_series.columns = ['ds', 'y']

    data_m.append(data_series)

    

comb_lst = [(id_lst[counter],data_m[counter]) for counter in range(0,len(id_lst))]
import time

start = time.time()

with Pool(4) as p:

    submission = p.starmap(run_prophet,comb_lst)



submission = pd.concat(submission,axis =0)

end = time.time()

elapsed_time = end-start

time_taken = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print('time',time_taken)
#temp = submission0.iloc[submission.shape[0]:,:]

#submission = pd.concat([submission,temp])
submission.shape

submission.to_csv('submission_evaluation.csv',index = False)

submission.shape