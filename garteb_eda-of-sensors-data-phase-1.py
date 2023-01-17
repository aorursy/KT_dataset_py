# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/sensor.csv', engine='python', sep = ',', decimal = '.')

df['timestamp'] =  pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S' ,utc='True')

df.head()

d = df['timestamp'].diff()

plt.plot(d[1:])
df.drop(['Unnamed: 0'], axis=1, inplace=True)


print(df.shape)

nan_stats = df.isnull().sum().sort_values(ascending = False)/df.shape[0]

nan_stats
df.drop(['sensor_15'], axis=1, inplace=True)
# plot in two windows with status - to asses nan's presence according to pump state.

# divide sesnro values by its max to get better scaling to status (actual values are not of our interest here)



fig, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1.set_ylabel(nan_stats.index[1])

ax1.plot(df['timestamp'],df['machine_status'])

ax1.plot(df['timestamp'],df[nan_stats.index[1]]/max(df[nan_stats.index[1]]))

ax2.set_ylabel(nan_stats.index[2])

ax2.plot(df['timestamp'],df['machine_status'])

ax2.plot(df['timestamp'],df[nan_stats.index[2]]/max(df[nan_stats.index[2]]))

ax3.set_ylabel(nan_stats.index[3])

ax3.plot(df['timestamp'],df['machine_status'])

ax3.plot(df['timestamp'],df[nan_stats.index[3]]/max(df[nan_stats.index[3]]))

plt.show()





fig, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1.set_ylabel(nan_stats.index[4])

ax1.plot(df['timestamp'],df['machine_status'])

ax1.plot(df['timestamp'],df[nan_stats.index[4]]/max(df[nan_stats.index[4]]))

ax2.set_ylabel(nan_stats.index[5])

ax2.plot(df['timestamp'],df['machine_status'])

ax2.plot(df['timestamp'],df[nan_stats.index[5]]/max(df[nan_stats.index[5]]))

ax3.set_ylabel(nan_stats.index[6])

ax3.plot(df['timestamp'],df['machine_status'])

ax3.plot(df['timestamp'],df[nan_stats.index[6]]/max(df[nan_stats.index[6]]))

plt.show()

df.drop(['sensor_50'], axis=1, inplace=True)
# features extractor for columns

def get_column_features(df):

    

    df_features = pd.DataFrame()

    col_list = ['Col','Max','Min','q_0.25','q_0.50','q_0.75']

    

    for column in df:

        tmp_q = df[column].quantile([0.25,0.5,0.75])

        tmp1 = pd.Series([column,max(df[column]),min(df[column]),tmp_q[0.25],tmp_q[0.5],tmp_q[0.75]],name='d')

        tmp1.index = col_list

        df_features = df_features.append(tmp1)

        

    return df_features



# get list os signal columns without timestamp and status:

signal_columns = [c for c in df.columns if c not in ['timestamp', 'machine_status']]

get_column_features(df[signal_columns])
corr = df[signal_columns].corr()

sns.heatmap(corr)
#locate indices of failure events and recovering and normal state

normal_idx = df.loc[df['machine_status'] == 'NORMAL'].index

failure_idx = df.loc[df['machine_status'] == 'BROKEN'].index

recovering_idx = df.loc[df['machine_status'] == 'RECOVERING'].index



bef_failure_idx = list()

for j in failure_idx:

    for i in range(24*60):

        bef_failure_idx.append(j-i)



bef_failure_idx.sort()



#locate timestamps of failures:

failures_timestamps = df.loc[failure_idx,'timestamp']

print(failures_timestamps)
for i in range(10):

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)

    

    ax1.set_ylabel(signal_columns[i*5])

    ax1.plot(df[signal_columns[i*5]],linewidth=1)

    ax1.plot(df.loc[list(recovering_idx),signal_columns[i*5]],linestyle='none',marker='.',color='red',markersize=4)

    ax1.plot(df.loc[bef_failure_idx,signal_columns[i*5]],linestyle='none',marker='.',color='orange',markersize=4)

        

    ax2.set_ylabel(signal_columns[i*5+1])

    ax2.plot(df[signal_columns[i*5+1]],linewidth=1)

    ax2.plot(df.loc[list(recovering_idx),signal_columns[i*5+1]],linestyle='none',marker='.',color='red',markersize=4)

    ax2.plot(df.loc[bef_failure_idx,signal_columns[i*5+1]],linestyle='none',marker='.',color='orange',markersize=4)

    

    ax3.set_ylabel(signal_columns[i*5+2])

    ax3.plot(df[signal_columns[i*5+2]],linewidth=1)

    ax3.plot(df.loc[list(recovering_idx),signal_columns[i*5+2]],linestyle='none',marker='.',color='red',markersize=4)

    ax3.plot(df.loc[bef_failure_idx,signal_columns[i*5+2]],linestyle='none',marker='.',color='orange',markersize=4)

    

    ax4.set_ylabel(signal_columns[i*5+3])

    ax4.plot(df[signal_columns[i*5+3]],linewidth=1)

    ax4.plot(df.loc[list(recovering_idx),signal_columns[i*5+3]],linestyle='none',marker='.',color='red',markersize=4)

    ax4.plot(df.loc[bef_failure_idx,signal_columns[i*5+3]],linestyle='none',marker='.',color='orange',markersize=4)

    

    ax5.set_ylabel(signal_columns[i*5+4])

    ax5.plot(df[signal_columns[i*5+4]],linewidth=1)

    ax5.plot(df.loc[list(recovering_idx),signal_columns[i*5+4]],linestyle='none',marker='.',color='red',markersize=4)

    ax5.plot(df.loc[bef_failure_idx,signal_columns[i*5+4]],linestyle='none',marker='.',color='orange',markersize=4)

    plt.show()


