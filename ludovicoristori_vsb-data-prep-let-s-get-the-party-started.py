import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline
import warnings

warnings.filterwarnings('ignore')
import gc

import pyarrow.parquet as pq
import os

print(os.listdir("../input"))
metadata_train = pd.read_csv("../input/metadata_train.csv")

metadata_train.info()
metadata_train.head(12)
metadata_test = pd.read_csv("../input/metadata_test.csv")

metadata_test.info()
metadata_test.head(12)
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.info()
sample_submission.head(12)
%%time

train = pq.read_pandas('../input/train.parquet').to_pandas()
train.info()
train.iloc[0:7,0:10]
x = train.index

y = train.iloc[:,0]
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_axes([0,0,1,1])

ax1.plot(x,y,color='lightblue')
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_axes([0,0,1,1])

ax1.set_xlim([10,300])

ax1.set_ylim([10,30])

ax1.plot(x,y,marker='o', color='orange')

ax2 = fig.add_axes([0.7,0.7,0.2,0.2])

ax2.plot(x,y, color='lightblue')
x = train.index

y0 = train.iloc[:,0]

y1 = train.iloc[:,1]

y2 = train.iloc[:,2]
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_axes([0,0,1,1])

ax1.plot(x,y0,color='blue')

ax1.plot(x,y1,color='red')

ax1.plot(x,y2,color='green')
np.mean(y0)
np.min(y0)
np.max(y0)
np.std(y0)
y0 = train.iloc[:,3]

y1 = train.iloc[:,4]

y2 = train.iloc[:,5]
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_axes([0,0,1,1])

ax1.plot(x,y0,color='blue')

ax1.plot(x,y1,color='red')

ax1.plot(x,y2,color='green')
np.mean(y0)
np.min(y0)
np.max(y0)
np.std(y0)
row_nr = train.shape[0]

row_nr
index_group_size=100
time_sample_idx=np.arange(0,row_nr,index_group_size)

time_sample_idx[0:10]
train_down=train.iloc[time_sample_idx,:]

train_down.iloc[:,0:10].head()
import numpy.fft as ft
def Amplitude(z):

    return np.abs(z)
def Phase(z):

    return (np.arctan2(z.imag,z.real))
df_harm=pd.DataFrame()
def find_dfa(df_source, df_dest,num_harm,base_col):

    # init

    dfa=df_dest.iloc[:,0:base_col]

    num_ap_cols = int(num_harm/2)

    for j in range(0,num_ap_cols) :

        dfa['Amp'+str(j)] = 0

        dfa['Pha'+str(j)] = 0

    dfa['ErrFun'] = 0

    dfa['ErrGen'] = 0

    # calc

    for i in range(0,len(df_source.columns)):

        dfa.loc[i]=0

        s=df_source.iloc[:,base_col+i]

        SF=ft.rfft(s)

        SF_Fundam=np.zeros(SF.size, dtype=np.complex_)

        SF_Filtered=np.zeros(SF.size, dtype=np.complex_)

        SF_Fundam[0:2]=SF[0:2]

        SF_Filtered[0:num_harm]=SF[0:num_harm]

        s_fun_rec=ft.irfft(SF_Fundam)

        s_gen_rec=ft.irfft(SF_Filtered)

        for j in range(0,num_ap_cols):

            dfa.iloc[i,base_col+2*j] = Amplitude(SF_Filtered[j])

            dfa.iloc[i,base_col+2*j+1] = Phase(SF_Filtered[j])

        dfa.iloc[i,base_col+2*num_ap_cols] = np.sqrt(np.mean((s-s_fun_rec)**2))

        dfa.iloc[i,base_col+2*num_ap_cols+1] = np.sqrt(np.mean((s-s_gen_rec)**2))

    return dfa
%%time

train_max = train.apply(np.max)

train_min = train.apply(np.min)
%%time

train_mean = train_down.apply(np.mean)

train_std = train_down.apply(np.std)
%%time

df_harm=pd.DataFrame()

num_harm=10

df_harm=find_dfa(train_down,df_harm,num_harm,0)
df_harm.iloc[:,0:10].head()
metadata_train['mean']=train_mean.values

metadata_train['max']=train_max.values

metadata_train['min']=train_min.values

metadata_train['std']=train_std.values

for j in range(0,int(num_harm/2)) :

    metadata_train['Amp'+str(j)] = df_harm['Amp'+str(j)]

    metadata_train['Pha'+str(j)] = df_harm['Pha'+str(j)]

metadata_train['ErrFun'] = df_harm['ErrFun']

metadata_train['ErrGen'] = df_harm['ErrGen']
metadata_train.head()
df_train = metadata_train

df_train.to_csv('df_train.csv', index=False)
col_group_size=2000
gc.collect()
metadata_test = pd.read_csv("../input/metadata_test.csv")
metadata_test['target']=-1

metadata_test['mean']=0

metadata_test['max']=0

metadata_test['min']=0

metadata_test['std']=0

for j in range(0,int(num_harm/2)) :

    metadata_test['Amp'+str(j)] = 0

    metadata_test['Pha'+str(j)] = 0

metadata_test['ErrFun'] = 0

metadata_test['ErrGen'] = 0
metadata_test.shape
metadata_test.head()
def add_info_test(metadata_df,time_sample_idx_1,col_group_size):

    col_id_start_0=np.min(metadata_test['signal_id'])

    col_id_start=col_id_start_0

    col_id_last=np.max(metadata_test['signal_id'])+1

    n_groups = int(np.round((col_id_last-col_id_start)/col_group_size))

    print('Steps = {}'.format(n_groups))

    for i in range(0,n_groups):

        col_id_stop = np.minimum(col_id_start+col_group_size,col_id_last)

        col_numbers = np.arange(col_id_start,col_id_stop)

        print('Step {s} - cols = [{a},{b})'.format(s=i,a=col_id_start,b=col_id_stop))

        print('   Adding Stats...',end="")

        col_names = [str(col_numbers[j]) for j in range(0,len(col_numbers))]

        test_i = pq.read_pandas('../input/test.parquet',columns=col_names).to_pandas()

        test_i_d1=test_i.iloc[time_sample_idx_1,:]

        test_mean_i = test_i_d1.apply(np.mean)

        test_max_i  = test_i.apply(np.max)

        test_min_i  = test_i.apply(np.min)

        test_std_i  = test_i_d1.apply(np.std)

        r_start = col_id_start - col_id_start_0

        r_stop = r_start + (col_id_stop-col_id_start)

        metadata_df.iloc[r_start:r_stop,4] = test_mean_i[0:col_id_stop-col_id_start].values

        metadata_df.iloc[r_start:r_stop,5] = test_max_i[0:col_id_stop-col_id_start].values

        metadata_df.iloc[r_start:r_stop,6] = test_min_i[0:col_id_stop-col_id_start].values

        metadata_df.iloc[r_start:r_stop,7] = test_std_i[0:col_id_stop-col_id_start].values

        print('   Adding FFT...')

        df_harm=pd.DataFrame()

        df_harm=find_dfa(test_i_d1,df_harm,10,0)

        num_ap_cols = int(num_harm/2)

        fft_base_col=8

        for j in range(0, num_ap_cols) :

            metadata_df.iloc[r_start:r_stop,fft_base_col+2*j] = df_harm.iloc[r_start:r_stop,2*j]

            metadata_df.iloc[r_start:r_stop,fft_base_col+2*j+1] = df_harm.iloc[r_start:r_stop,2*j+1]

        metadata_df.iloc[r_start:r_stop,fft_base_col+num_harm] = df_harm.iloc[r_start:r_stop, num_harm]

        metadata_df.iloc[r_start:r_stop,fft_base_col+num_harm+1] = df_harm.iloc[r_start:r_stop, num_harm+1]

        col_id_start=col_id_stop

    return (metadata_df)
%%time

metadata_test1=add_info_test(metadata_test,time_sample_idx,col_group_size)
metadata_test1.head()
metadata_test1.shape
metadata_test1.iloc[0:12,:]
metadata_test1.to_csv('df_test.csv', index=False)