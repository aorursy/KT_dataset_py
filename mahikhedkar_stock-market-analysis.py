import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
amd = pd.read_csv('../input/intel-stock-market/INTC (1).csv')
amd = pd.read_csv('../input/intel-stock-market/INTC (1).csv',header=0,index_col='Date',parse_dates=True)
amd.head(5)
amd.tail()
import pandas_datareader as pdr
import datetime
nvda = pdr.get_data_yahoo('NVDA',
                         start = datetime.datetime(2004,1,1),
                         end = datetime.datetime(2019,9,15))
qcom = pdr.get_data_yahoo('QCOM',
                         start = datetime.datetime(2004,1,1),
                         end = datetime.datetime(2019,9,15))
intc = pdr.get_data_yahoo('INTC',
                         start = datetime.datetime(2004,1,1),
                         end = datetime.datetime(2019,9,15))
ibm = pdr.get_data_yahoo('IBM',
                        start = datetime.datetime(2004,1,1),
                        end = datetime.datetime(2019,9,15))
type(nvda),type(amd)
nvda.head()
ibm.tail()
ibm.describe()
nvda.columns
nvda.index,amd.index
nvda.shape

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.dates as mdates
plt.plot(ibm.index,ibm['Adj Close'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.grid(True)
plt.xticks(rotation = 90)
plt.show()
#Subplots
f,ax = plt.subplots(2, 2,figsize=(10,10), sharex = True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
f.gca().xaxis.set_major_locator(mdates.YearLocator())

ax[0,0].plot(nvda.index, nvda['Adj Close'],color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation = 90)
ax[0,0].set_title('NVIDIA');

ax[0,1].plot(intc.index, intc['Adj Close'],color='b')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation = 90)
ax[0,1].set_title('INTEL');

ax[1,0].plot(qcom.index, qcom['Adj Close'],color='g')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation = 90)
ax[1,0].set_title('QUALCOMM');

ax[1,1].plot(amd.index,amd['Adj Close'],color = 'y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation = 90)
ax[1,1].set_title('AMD');

#plt.show()
ibm_2018 = ibm.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]
plt.plot(ibm_2018.index, ibm_2018['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation = 90)
plt.show()
#Subplots
f,ax = plt.subplots(2,2, figsize = (10,10),sharex=True,sharey=True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
f.gca().xaxis.set_major_locator(mdates.MonthLocator())
 
nvda_2018 = nvda.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31') ]
ax[0,0].plot(nvda_2018.index, nvda_2018['Adj Close'],'.',color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

intc_2018 = intc.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[0,1].plot(intc_2018.index, intc_2018['Adj Close'],'.',color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

qcom_2018 = qcom.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,0].plot(qcom_2018.index,qcom_2018['Adj Close'],'.',color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

amd_2018 = amd.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,1].plot(amd_2018.index, amd_2018['Adj Close'],'.',color='y')
ax[1,1].grid(True) 
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD')                                       
                                                    
monthly_nvda_2018 = nvda_2018.resample('4M').mean()
plt.scatter(monthly_nvda_2018.index, monthly_nvda_2018['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()
#Subplots
f, ax = plt.subplots(2,2, figsize=(10,10),sharex=True ,sharey =True)

monthly_nvda_2018 = nvda_2018.resample('4M').mean()
ax[0,0].scatter(monthly_nvda_2018.index,monthly_nvda_2018['Adj Close'],color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

monthly_intc_2018 = intc_2018.resample('4M').mean()
ax[0,1].scatter(monthly_intc_2018.index,monthly_intc_2018['Adj Close'],color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL')

monthly_qcom_2018 = qcom_2018.resample('4M').mean()
ax[1,0].scatter(monthly_qcom_2018.index,monthly_qcom_2018['Adj Close'], color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation =90)
ax[1,0].set_title('QUALCOMM');

monthly_amd_2018 = amd_2018.resample('4M').mean()
ax[1,1].scatter(monthly_amd_2018.index,monthly_amd_2018['Adj Close'], color='y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');

ibm_19 = ibm.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]

weekly_ibm_19 = ibm_19.resample('W').mean()
weekly_ibm_19.head()
plt.plot(weekly_ibm_19.index, weekly_ibm_19['Adj Close'], '-o')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()

#Subplots
nvda_19 = nvda.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_nvda_19 = nvda_19.resample('W').mean()

intc_19 = intc.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_intc_19 = intc_19.resample('W').mean()

qcom_19 = qcom.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_qcom_19 = qcom_19.resample('W').mean()

amd_19 = amd.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_amd_19 = amd_19.resample('W').mean()

f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
ax[0,0].plot(weekly_nvda_19.index, weekly_nvda_19['Adj Close'], '-o', color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

ax[0,1].plot(weekly_intc_19.index, weekly_intc_19['Adj Close'], '-o',color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

ax[1,0].plot(weekly_qcom_19.index, weekly_qcom_19['Adj Close'],'-o', color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

ax[1,1].plot(weekly_amd_19.index, weekly_amd_19['Adj Close'], '-o', color='y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');
ibm['diff'] = ibm['Open'] - ibm['Close']
ibm_diff = ibm.resample('W').mean()
ibm_diff.tail(10)
plt.scatter(ibm_diff.loc['2019-01-01':'2019-09-15'].index, ibm_diff.loc['2019-01-01':'2019-09-15']['diff'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()

#Subplots
nvda['diff'] = nvda['Open'] - nvda['Close']
nvda_diff = nvda.resample('W').mean()

intc['diff'] = intc['Open'] - intc['Close']
intc_diff = intc.resample('W').mean()

qcom['diff'] = qcom['Open'] - qcom['Close']
qcom_diff = qcom.resample('W').mean()

amd['diff'] = amd['Open'] - amd['Close']
amd_diff = amd.resample('W').mean()

f, ax = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

ax[0,0].scatter(nvda_diff.loc['2019-01-01':'2019-09-15'].index, nvda_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

ax[0,1].scatter(intc_diff.loc['2019-01-01':'2019-09-15'].index, intc_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

ax[1,0].scatter(qcom_diff.loc['2019-01-01':'2019-09-15'].index, qcom_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

ax[1,1].scatter(amd_diff.loc['2019-01-01':'2019-09-15'].index, amd_diff.loc['2019-01-01':'2019-09-15']['diff']
, color='y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');

daily_close_ibm = ibm[['Adj Close']]

# Daily returns
daily_pct_change_ibm = daily_close_ibm.pct_change()

# Replace NA values with 0
daily_pct_change_ibm.fillna(0, inplace=True)

daily_pct_change_ibm.head()
daily_pct_change_ibm.hist(bins=50)

# Show the plot
plt.show()
daily_close_nvda = nvda[['Adj Close']]

# Daily returns
daily_pct_change_nvda = daily_close_nvda.pct_change()

# Replace NA values with 0
daily_pct_change_nvda.fillna(0, inplace=True)

daily_close_intc = intc[['Adj Close']]

# Daily returns
daily_pct_change_intc = daily_close_intc.pct_change()

# Replace NA values with 0
daily_pct_change_intc.fillna(0, inplace=True)

daily_close_qcom = qcom[['Adj Close']]

# Daily returns
daily_pct_change_qcom = daily_close_qcom.pct_change()

# Replace NA values with 0
daily_pct_change_qcom.fillna(0, inplace=True)

daily_close_amd = amd[['Adj Close']]

# Daily returns
daily_pct_change_amd = daily_close_amd.pct_change()

# Replace NA values with 0
daily_pct_change_amd.fillna(0, inplace=True)

daily_pct_change_amd.head()
import seaborn as sns
sns.set()
import seaborn as sns
# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(12, 7))

# Plot a simple histogram with binsize determined automatically
sns.distplot(daily_pct_change_nvda['Adj Close'], color="b", ax=axes[0, 0], axlabel='NVIDIA');

# Plot a kernel density estimate and rug plot
sns.distplot(daily_pct_change_intc['Adj Close'], color="r", ax=axes[0, 1], axlabel='INTEL');

# Plot a filled kernel density estimate
sns.distplot(daily_pct_change_qcom['Adj Close'], color="g", ax=axes[1, 0], axlabel='QUALCOMM');

# Plot a historgram and kernel density estimate
sns.distplot(daily_pct_change_amd['Adj Close'], color="m", ax=axes[1, 1], axlabel='AMD');
import numpy as np
min_periods = 75 

# Calculate the volatility
vol = daily_pct_change_ibm.rolling(min_periods).std() * np.sqrt(min_periods) 

vol.fillna(0,inplace=True)

vol.tail()
# Plot the volatility
vol.plot(figsize=(10, 8))

# Show the plot
plt.show()
ibm_adj_close_px = ibm['Adj Close']
# Short moving window rolling mean
ibm['42'] = ibm_adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
ibm['252'] = ibm_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
ibm[['Adj Close', '42', '252']].plot(title="IBM")

# Show plot
plt.show()

nvda_adj_close_px = nvda['Adj Close']
# Short moving window rolling mean
nvda['42'] = nvda_adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
nvda['252'] = nvda_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
nvda[['Adj Close', '42', '252']].plot(title="NVIDIA")

# Show plot
plt.show()
ibm.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="IBM in 2019");
nvda.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="NVIDIA in 2019");
