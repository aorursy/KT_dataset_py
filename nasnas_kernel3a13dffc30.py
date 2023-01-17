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

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean



from sklearn.metrics import mean_absolute_error



warnings.filterwarnings('ignore')

%matplotlib inline



#Будем вычислять RMSE

from sklearn.metrics import mean_squared_error

from math import sqrt

    

def rmse(y_actual, y_predicted):

    rmse = sqrt(mean_squared_error(y_actual, y_predicted))

    return rmse



# Относительная ошибка - хорошая метрика для бизнеса

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
raw_data = pd.read_csv('/kaggle/input/sputnik/train.csv', sep =',')

raw_data.head(2)
def make_coord(df,coord,coord_sim):

    ts_x=df[coord]

    ts_x_sim = df[coord_sim]

    

    eps = np.mean(df.epoch-df.epoch.shift(1))/5

    bad_num_plus=[]

    [bad_num_plus.append(i) for i in range(1,df.shape[0]) if (df.epoch[i]-df.epoch[i-1])<eps]

    

    ts_x.drop(ts_x.index[bad_num_plus], inplace=True)

    if len(bad_num_plus)>0:

        val_x=ts_x_sim.values[:-len(bad_num_plus)]

    else: 

        val_x=ts_x_sim.values

    ts_x_sim = pd.Series(val_x)

    ts_x_sim.index=ts_x.index

    

    d_x = ts_x - ts_x_sim

    #size = len(d_x)//3

    train = d_x[df.type=='train']

    val   = d_x[df.type=='test']

    

    fit1 = ExponentialSmoothing(np.asarray(train), seasonal_periods=24, seasonal='add').fit()

    forecast = pd.Series(fit1.forecast(len(val)))

    forecast.index = val.index

    

    c=1.1

    x_pd = train.append(c*forecast)

    x_pd = x_pd + ts_x_sim

    

    x_new=np.array(x_pd.values)

    bad_num=np.array(bad_num_plus)-1

    for i in bad_num:

        x_new = np.insert(x_new,i,x_new[i])

    x_new = pd.Series(x_new)

    x_new.index = df.index

    

    df[coord+'_new']=x_new

    return df   
def make_coord(df,coord,coord_sim):

    ts_x=df[coord]

    ts_x_sim = df[coord_sim]

    

    eps = np.mean(df.epoch-df.epoch.shift(1))/5

    bad_num_plus=[]

    [bad_num_plus.append(i) for i in range(1,df.shape[0]) if (df.epoch[i]-df.epoch[i-1])<eps]

    

    ts_x.drop(ts_x.index[bad_num_plus], inplace=True)

    if len(bad_num_plus)>0:

        val_x=ts_x_sim.values[:-len(bad_num_plus)]

    else: 

        val_x=ts_x_sim.values

    ts_x_sim = pd.Series(val_x)

    ts_x_sim.index=ts_x.index

    

    d_x = ts_x - ts_x_sim

    #size = len(d_x)//3

    train = d_x[df.type=='train']

    val   = d_x[df.type=='test']

    

    fit1 = ExponentialSmoothing(np.asarray(train), seasonal_periods=24, seasonal='add').fit()

    forecast = pd.Series(fit1.forecast(len(val)))

    forecast.index = val.index

    

    c_1=(max(train[-24:])-np.mean(train[-24:]))/(max(train[-48:-24])-np.mean(train[-24:]))

    c_2=(min(train[-24:])-np.mean(train[-24:]))/(min(train[-48:-24])-np.mean(train[-24:]))

    

    for j in range(len(forecast)):

        if forecast[j]>np.mean(forecast[-48:]):

            forecast[j]= np.mean(forecast[-48:]) + (forecast[j]-np.mean(forecast[-48:]))*(c_1**(1+j/24))

        else:

            forecast[j]= np.mean(forecast[-48:]) + (forecast[j]-np.mean(forecast[-48:]))*(c_2**(1+j/24))



    x_pd = train.append(forecast)

    x_pd = x_pd + ts_x_sim

    

    x_new=np.array(x_pd.values)

    bad_num=np.array(bad_num_plus)-1

    for i in bad_num:

        x_new = np.insert(x_new,i,x_new[i])

    x_new = pd.Series(x_new)

    x_new.index = df.index

    

    df[coord+'_new']=x_new

    return df
raw_train = raw_data #сначала я тренировался похожим способом



ids=[]

n_err=[]

err_by_obj=[]



for i in range(600):

    #print(i)

    df = raw_train[raw_train.sat_id==i]

    df['error']  = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)



    df.sort_values('epoch', axis = 0, inplace=True)

    df.epoch = pd.to_datetime(df.epoch, format='%Y-%m-%dT%H:%M:%S')

    df.index = df.epoch

    

    df = make_coord(df,'x','x_sim')

    df = make_coord(df,'y','y_sim')

    df = make_coord(df,'z','z_sim')

    

    df['new_error']  = np.linalg.norm(df[['x_new', 'y_new', 'z_new']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

    for j in range(df.shape[0]):

        ids.append(df.id[j])

        n_err.append(df.new_error[j])

        

    #err_by_obj.append(mean_absolute_percentage_error(df.error[-df.shape[0]//3:], df.new_error[-df.shape[0]//3:]))

    
raw_train['new_error']=n_err

raw_train.head(2)
raw_train[raw_train.type=='test'].tail(2)
res =  raw_train[['id','new_error','type']]

res.columns = ['id', 'error','type']

res=res[res.type=='test']

res=res[['id','error']]

res.to_csv('my_sub_sec.csv', index = False, header=True)