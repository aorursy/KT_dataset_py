# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
amd=pd.read_csv('/kaggle/input/INTC (1).csv',header=0,index_col='Date',parse_dates=True)
amd.head(10)
import pandas_datareader as pdr
import datetime
nvda=pdr.get_data_yahoo('NVDA',
                       start=datetime.datetime(2000,1,1),
                       end=datetime.datetime(2020,4,26))
qcom=pdr.get_data_yahoo('QCOM',
                       start=datetime.datetime(2000,1,1),
                       end=datetime.datetime(2020,4,26))
intc=pdr.get_data_yahoo('INTC',
                       start=datetime.datetime(2000,1,1),
                       end=datetime.datetime(2020,4,26))
ibm=pdr.get_data_yahoo('IBM',
                       start=datetime.datetime(2000,1,1),
                       end=datetime.datetime(2020,4,26))
type(nvda)
nvda.head(n=2)
ibm.tail()
ibm.describe()
nvda.columns
nvda.index,amd.index
nvda.shape
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.dates as mdates
plt.figure(figsize=(10,8))
plt.plot(ibm.index,ibm['Adj Close'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
f,ax=plt.subplots(2,2,figsize=(15,15),sharex=True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
f.gca().xaxis.set_major_locator(mdates.YearLocator())
ax[0,0].plot(nvda.index,nvda['Adj Close'],color='indigo')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

ax[0,1].plot(intc.index,intc['Adj Close'],color='teal')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

ax[1,0].plot(qcom.index,qcom['Adj Close'],color='cyan')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUAL COM');

ax[1,1].plot(amd.index,amd['Adj Close'],color='gray')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');
ibm_19=ibm.loc[pd.Timestamp('2019-01-01'):pd.Timestamp('2019-12-31')]
plt.plot(ibm_19.index,ibm_19['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
#downsampling
f,ax=plt.subplots(2,2,figsize=(6,6),sharex=True,sharey=True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
nvda_19=nvda.loc[pd.Timestamp('2019-01-01'):pd.Timestamp('2019-12-31')]
ax[0,0].plot(nvda_19.index,nvda_19['Adj Close'],',',color='indigo')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');
intc_19=intc.loc[pd.Timestamp('2019-01-01'):pd.Timestamp('2019-12-31')]
ax[0,1].plot(intc_19.index,intc_19['Adj Close'],',',color='teal')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');
qcom_19=qcom.loc[pd.Timestamp('2019-01-01'):pd.Timestamp('2019-12-31')]
ax[1,0].plot(qcom_19.index,qcom_19['Adj Close'],',',color='cyan')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUAL COMM');
amd_19=amd.loc[pd.Timestamp('2019-01-01'):pd.Timestamp('2019-12-31')]
ax[1,1].plot(amd_19.index,amd_19['Adj Close'],',',color='gray')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');


monthly_nvda_19=nvda_19.resample('4M').mean()
plt.scatter(monthly_nvda_19.index,monthly_nvda_19['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
f,ax= plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
monthly_nvda_19=nvda_19.resample('4M').mean()
ax[0,0].scatter(monthly_nvda_19.index,monthly_nvda_19['Adj Close'],color='indigo')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');
monthly_intc_19=intc_19.resample('4M').mean()
ax[0,1].scatter(monthly_intc_19.index,monthly_intc_19['Adj Close'],color='teal')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');
monthly_qcom_19=qcom_19.resample('4M').mean()
ax[1,0].scatter(monthly_qcom_19.index,monthly_qcom_19['Adj Close'],color='cyan')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUAL COMM');
monthly_amd_19=amd_19.resample('4M').mean()
ax[1,1].scatter(monthly_amd_19.index,monthly_amd_19['Adj Close'],color='gray')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');
ibm_20=ibm.loc[pd.Timestamp('2020-01-01'):pd.Timestamp('2020-04-27')]
w_ibm_20=ibm_20.resample('W').mean()
w_ibm_20.head()
plt.plot(w_ibm_20.index,w_ibm_20['Adj Close'],'-o')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
nvda_20=nvda.loc[pd.Timestamp('2020-01-01'):pd.Timestamp('2020-04-27')]
w_nvda_20=nvda_20.resample('W').mean()
intc_20=intc.loc[pd.Timestamp('2020-01-01'):pd.Timestamp('2020-04-27')]
w_intc_20=intc_20.resample('W').mean()
qcom_20=qcom.loc[pd.Timestamp('2020-01-01'):pd.Timestamp('2020-04-27')]
w_qcom_20=qcom_20.resample('W').mean()
amd_20=amd.loc[pd.Timestamp('2020-01-01'):pd.Timestamp('2020-04-27')]
w_amd_20=amd_20.resample('W').mean()
f,ax= plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
ax[0,0].plot(w_nvda_20.index,w_nvda_20['Adj Close'],'-o',color='indigo')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');
ax[0,1].plot(w_intc_20.index,w_intc_20['Adj Close'],'-o',color='teal')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');
ax[1,0].plot(w_qcom_20.index,w_qcom_20['Adj Close'],'-o',color='cyan')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUAL COMM');
ax[1,1].plot(w_amd_20.index,w_amd_20['Adj Close'],'-o',color='gray')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');

ibm['diff']=ibm['Open']-ibm['Close']
ibm_diff=ibm.resample('W').mean()
ibm_diff.tail(10)
plt.scatter(ibm_diff.loc['2020-01-01':'2020-04-27'].index,ibm_diff.loc['2020-01-01':'2020-04-27']['diff'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
nvda['diff']=nvda['Open']-nvda['Close']
nvda_diff=nvda.resample('W').mean()
plt.scatter(nvda_diff.loc['2020-01-01':'2020-04-27'].index,nvda_diff.loc['2020-01-01':'2020-04-27']['diff'],color='indigo')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
intc['diff']=intc['Open']-intc['Close']
intc_diff=intc.resample('W').mean()
plt.scatter(intc_diff.loc['2020-01-01':'2020-04-27'].index,intc_diff.loc['2020-01-01':'2020-04-27']['diff'],color='teal')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
qcom['diff']=qcom['Open']-qcom['Close']
qcom_diff=qcom.resample('W').mean()
plt.scatter(qcom_diff.loc['2020-01-01':'2020-04-27'].index,qcom_diff.loc['2020-01-01':'2020-04-27']['diff'],color='cyan')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
amd['diff']=amd['Open']-amd['Close']
amd_diff=amd.resample('W').mean()
plt.scatter(amd_diff.loc['2020-01-01':'2020-04-27'].index,amd_diff.loc['2020-01-01':'2020-04-27']['diff'],color='gray')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
daily_close_ibm=ibm[['Adj Close']]
#daily return
daily_pct_change_ibm=daily_close_ibm.pct_change()
daily_pct_change_ibm.fillna(0,inplace=True)
daily_pct_change_ibm.head()
daily_pct_change_ibm.hist(bins=50)
plt.show()
daily_close_nvda=nvda[['Adj Close']]
daily_pct_change_nvda=daily_close_nvda.pct_change()
daily_pct_change_nvda.fillna(0,inplace=True)

daily_close_intc=intc[['Adj Close']]
daily_pct_change_intc=daily_close_intc.pct_change()
daily_pct_change_intc.fillna(0,inplace=True)

daily_close_qcom=qcom[['Adj Close']]
daily_pct_change_qcom=daily_close_qcom.pct_change()
daily_pct_change_qcom.fillna(0,inplace=True)

daily_close_amd=amd[['Adj Close']]
daily_pct_change_amd=daily_close_amd.pct_change()
daily_pct_change_amd.fillna(0,inplace=True)
daily_pct_change_amd.head()
import seaborn as sns
sns.set()
import seaborn as sns
f,axes=plt.subplots(2,2,figsize=(12,7))
sns.distplot(daily_pct_change_nvda['Adj Close'],color='indigo',ax=axes[0,0],axlabel='NVIDIA');
sns.distplot(daily_pct_change_intc['Adj Close'],color='teal',ax=axes[0,1],axlabel='INTEL');
sns.distplot(daily_pct_change_qcom['Adj Close'],color='cyan',ax=axes[1,0],axlabel='QUAL COMM');
sns.distplot(daily_pct_change_amd['Adj Close'],color='gray',ax=axes[1,1],axlabel='AMD');
import numpy as np
min_periods=75
#calculating volatility
vol=daily_pct_change_ibm.rolling(min_periods).std()*np.sqrt(min_periods)
vol.fillna(0,inplace=True)
vol.tail()
vol.plot(figsize=(10,8))
plt.show()
ibm_adj_close_px=ibm['Adj Close']
ibm['42']=ibm_adj_close_px.rolling(window=40).mean()
ibm['252']=ibm_adj_close_px.rolling(window=252).mean()
ibm[['Adj Close','42','252']].plot(title="IBM")
plt.show()
nvda_adj_close_px=nvda['Adj Close']
nvda['42']=nvda_adj_close_px.rolling(window=40).mean()
nvda['252']=nvda_adj_close_px.rolling(window=252).mean()
nvda[['Adj Close','42','252']].plot(title="NVIDIA")
plt.show()
intc_adj_close_px=intc['Adj Close']
intc['42']=intc_adj_close_px.rolling(window=40).mean()
intc['252']=intc_adj_close_px.rolling(window=252).mean()
intc[['Adj Close','42','252']].plot(title="INTEL")
plt.show()
qcom_adj_close_px=qcom['Adj Close']
qcom['42']=qcom_adj_close_px.rolling(window=40).mean()
qcom['252']=qcom_adj_close_px.rolling(window=252).mean()
qcom[['Adj Close','42','252']].plot(title="QUAL COMM")
plt.show()
amd_adj_close_px=amd['Adj Close']
amd['42']=amd_adj_close_px.rolling(window=40).mean()
amd['252']=amd_adj_close_px.rolling(window=252).mean()
amd[['Adj Close','42','252']].plot(title="AMD")
plt.show()


ibm.loc['2020-01-01':'2020-04-27'][['Adj Close','42','252']].plot(title="IBM in 2020");
nvda.loc['2020-01-01':'2020-04-27'][['Adj Close','42','252']].plot(title="NVDA in 2020");
intc.loc['2020-01-01':'2020-04-27'][['Adj Close','42','252']].plot(title="INTEL in 2020");
qcom.loc['2020-01-01':'2020-04-27'][['Adj Close','42','252']].plot(title="QUAL COMM in 2020");
amd.loc['2020-01-01':'2020-04-27'][['Adj Close','42','252']].plot(title="AMD in 2020");

















































