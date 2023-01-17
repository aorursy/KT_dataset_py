import numpy as np

import os

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

from datetime import datetime



#For faster compute

from numba import jit, cuda

from numba import float64, int64, prange



#For Data Processing

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.autograd import Variable

from torch.utils.data.dataset import Dataset

from torch.utils.data.sampler import SubsetRandomSampler
files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))

files.sort()

train_files = files[3:4]

test_files = files[1:6]

val_files = files[-1]
def read_files(folder):

    df = [pd.read_csv(f) for f in folder]

    big_frame = pd.concat(df, ignore_index=True)

    big_frame = big_frame[big_frame.symbol =='XBTUSD']

    big_frame['timestamp'] = big_frame.timestamp.map(lambda t: datetime.strptime(t[:-3], '%Y-%m-%dD%H:%M:%S.%f'))

    return big_frame
train_df = read_files(train_files)

train_df.head()
'''

Volume weighted average price: tracked for specific interval

'''

def compute_vwap(df):

    weights = df['foreignNotional']

    prices = df['price']

    vwap = np.sum(weights*prices) // np.sum(weights)

    df['vwap'] = vwap

    return df



#Check on train data

data = train_df



data_timeidx = data.set_index('timestamp')

data_time_grp = data_timeidx.groupby(pd.Grouper(freq='15Min'))

num_time_bars = len(data_time_grp) 

data_time_vwap = data_time_grp.apply(compute_vwap)

data_time_vwap['vwap'].plot(label='Time', figsize=(15,5))
total_ticks = len(data)

#round to the nearest thousand

num_ticks_per_bar = round(total_ticks / num_time_bars, -3)

data_tick_grp = data.reset_index().assign(grpId=lambda row: row.index // num_ticks_per_bar)

data_tick_vwap =  data_tick_grp.groupby('grpId').apply(compute_vwap)

data_tick_vwap.set_index('timestamp', inplace=True)

data_time_vwap['vwap'].plot(label='Time', figsize=(15,5))

data_tick_vwap['vwap'].plot(label='Tick', figsize=(15,5))

plt.legend();
data_cm_vol = data.assign(cmVol=data['homeNotional'].cumsum()) 

total_vol = data_cm_vol.cmVol.values[-1]

vol_per_bar = total_vol / num_time_bars

vol_per_bar = round(vol_per_bar, -3) # round to the nearest thousand

data_vol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)

data_vol_vwap =  data_vol_grp.groupby('grpId').apply(compute_vwap)

data_vol_vwap.set_index('timestamp', inplace=True)

data_time_vwap['vwap'].plot(label='Time', figsize=(15,5))

data_vol_vwap['vwap'].plot(label='Tick', figsize=(15,5))
data_cm_dol = data.assign(cmVol=data['foreignNotional'].cumsum()) 

total_dol = data_cm_vol.cmVol.values[-1]

dol_per_bar = total_vol / num_time_bars

dol_per_bar = round(vol_per_bar, -3) # round to the nearest thousand

data_dol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)

data_dol_vwap =  data_vol_grp.groupby('grpId').apply(compute_vwap)

data_dol_vwap.set_index('timestamp', inplace=True)

data_time_vwap['vwap'].plot(label='Time', figsize=(15,5))

data_dol_vwap['vwap'].plot(label='Tick', figsize=(15,5))
def convert_tick_direction(tick_direction):

    if tick_direction in ('PlusTick', 'ZeroPlusTick'):

        return 1

    elif tick_direction in ('MinusTick', 'ZeroMinusTick'):

        return -1

    else:

        raise ValueError('converting invalid input: '+ str(tick_direction))

data_timeidx['tickDirection'] = data_timeidx.tickDirection.map(convert_tick_direction)
data_signed_flow = data_timeidx.assign(bv = data_timeidx.tickDirection * data_timeidx.foreignNotional)
@jit((float64[:], int64), nopython=True, nogil=True,) #compiler options for function input and its runtime feature

def _ewma(arr_in, window):

    n = arr_in.shape[0]

    ewma = np.empty(n, dtype=float64)

    alpha = 2 / float(window + 1)

    w = 1

    ewma_old = arr_in[0]

    ewma[0] = ewma_old

    for i in prange(1, n):

        w += (1-alpha)**i

        ewma_old = ewma_old*(1-alpha) + arr_in[i]

        ewma[i] = ewma_old / w

    return ewma
abs_Ebv_init = np.abs(data_signed_flow['bv'].mean()) #Use the mean value as its close to expected bv

E_T_init = 500000 # 500000 ticks to warm up



def compute_Ts(bvs, E_T_init, abs_Ebv_init):

    #Tickets and their positions

    Ts, i_s = [], []

    i_prev, E_T, abs_Ebv  = 0, E_T_init, abs_Ebv_init

    

    n = bvs.shape[0]

    bvs_val = bvs.values.astype(np.float64)

    abs_thetas, thresholds = np.zeros(n), np.zeros(n)

    abs_thetas[0], cur_theta = np.abs(bvs_val[0]), bvs_val[0]

    for i in prange(1, n):

        cur_theta += bvs_val[i]

        abs_theta = np.abs(cur_theta)

        abs_thetas[i] = abs_theta

        

        threshold = E_T * abs_Ebv

        thresholds[i] = threshold

        if abs_theta >= threshold:

            cur_theta = 0

            #Tick length as distance b/w bar indexes

            Ts.append(np.float64(i - i_prev)) 

            i_s.append(i)

            i_prev = i



            E_T = _ewma(np.array(Ts), window=np.int64(len(Ts)))[-1]

            abs_Ebv = np.abs( _ewma(bvs_val[:i], window=np.int64(E_T_init * 3))[-1] ) # window of 3 bars

    return Ts, abs_thetas, thresholds, i_s

Ts, abs_thetas, thresholds, i_s = compute_Ts(data_signed_flow.bv, E_T_init, abs_Ebv_init)
n = data_signed_flow.shape[0]

i_iter = iter(i_s + [n])

i_cur = i_iter.__next__()

grpId = np.zeros(n)

for i in range(1, n):

    if i <= i_cur:

        grpId[i] = grpId[i-1]

    else:

        grpId[i] = grpId[i-1] + 1

        i_cur = i_iter.__next__()
data_dollar_imb_grp = data_signed_flow.assign(grpId = grpId)

data_dollar_imb_vwap = data_dollar_imb_grp.groupby('grpId').apply(compute_vwap)

data_time_vwap['vwap'].plot(label='Time', figsize=(15,5))

data_dollar_imb_vwap['vwap'].plot(label='Tick', figsize=(15,5))

plt.legend();