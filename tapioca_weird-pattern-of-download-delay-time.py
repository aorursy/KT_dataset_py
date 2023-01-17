import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import os
import gc
#print(os.listdir("../input"))

import time

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#make wider graphs
sns.set(rc={'figure.figsize':(12,5)});
plt.figure(figsize=(12,5));
#import first24-hour of train data
train = pd.read_csv('../input/train.csv', parse_dates=['click_time','attributed_time'], nrows=55442161, usecols=['app','click_time', 'attributed_time','is_attributed'])
#set click_time and attributed_time as timeseries
download = train[train['is_attributed']==1]
download['click_time'] = pd.to_datetime(download['click_time'])
download['attributed_time'] = pd.to_datetime(download['attributed_time'])
download['download_delay_time'] = (download['attributed_time'] - download['click_time']).dt.total_seconds() / 3600
#double check that 'attributed_time' is not Null for all values that resulted in download (i.e. is_attributed == 1)
download['click_hour'] = pd.to_datetime(download.click_time).dt.hour.astype('uint8')
download['attributed_hour'] = pd.to_datetime(download.attributed_time).dt.hour.astype('uint8')
sns.barplot('click_hour','download_delay_time', data=download)
sns.barplot('attributed_hour','download_delay_time', data=download)
feature = 'app'
gk = train[[feature,'is_attributed']].groupby(by=[feature])[['is_attributed']].mean().reset_index().rename(index=str, columns={'is_attributed': 'download_rate'})
gp = download[[feature,'download_delay_time']].groupby(by=[feature])[['download_delay_time']].mean().reset_index().rename(index=str, columns={'download_delay_time': 'mean_download_delay_time'})
merge = gp.merge(gk, on=[feature], how='left')
sns.lmplot(x="mean_download_delay_time", y="download_rate", data=merge)
plt.title('Download-rate vs. Download_delay_time')
plt.ylim(0,1)
plt.xlim(0,24)
gp = gp.sort_values(by='mean_download_delay_time',ascending=True).reset_index()
gp.drop('index',axis=1,inplace=True)
print(gp.head())
print(gp.tail())
#20 apps with longest download time
sns.barplot(feature,'mean_download_delay_time', data=gp.tail(20))
#20 apps with shorest download time
sns.barplot(feature,'mean_download_delay_time', data=gp.head(20))
del train, download, gk, gp
gc.collect()
train = pd.read_csv('../input/train.csv', parse_dates=['click_time','attributed_time'],skiprows=list(range(1,55442161)),nrows=61618817, usecols=['click_time', 'attributed_time','is_attributed'])
download = train[train['is_attributed']==1]
download['click_time'] = pd.to_datetime(download['click_time'])
download['attributed_time'] = pd.to_datetime(download['attributed_time'])
download['download_delay_time'] = (download['attributed_time'] - download['click_time']).dt.total_seconds() / 3600
download['click_hour'] = pd.to_datetime(download.click_time).dt.hour.astype('uint8')
sns.barplot('click_hour','download_delay_time', data=download)