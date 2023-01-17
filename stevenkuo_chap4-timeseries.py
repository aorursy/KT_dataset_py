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
print(os.listdir("../input/web-traffic-time-series-forecasting"))
!kaggle ls
train = pd.read_csv('../input/web-traffic-time-series-forecasting/train_1.csv.zip').fillna(0)
train.head()
def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]),x[-3],x[-2],x[-1]
parse_page(train.Page[0])
l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
df.columns = ['Subject', 'Sub_Page','Access','Agent']
train = pd.concat([train,df],axis=1)
del train['Page']
train.head()
import matplotlib.pyplot as plt
import re
%matplotlib inline
fig , ax = plt.subplots(figsize=(10,7))
train.Sub_Page.value_counts().plot(kind='bar',)
train.Access.value_counts().plot(kind='bar')
train.Agent.value_counts().plot(kind='bar')
idx = 39457
window = 10
data = train.iloc[idx,0:-4]
name = train.iloc[idx,-4]
days = [r for r in range(data.shape[0])]
fig , ax = plt.subplots(figsize=(16,7))
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title(name)

ax.plot(days,data.values, color='grey')
ax.plot(np.convolve(data,
                   np.ones((window,))/window,
                   mode='valid'), color='black')
ax.set_yscale('log')
data[:5]
fig, ax = plt.subplots(figsize=(10,7))
plt.ylabel('View per Page')
plt.xlabel('Day')
plt.title('Twenty One Pilots Popularity')
ax.set_yscale('log')
handles = []
for country in ['de','en','es','fr','ru']:
    idx = np.where((train['Subject'] == 'Twenty One Pilots')
                  & (train['Sub_Page'] == '{}.wikipedia.org'.format(country))
                  & (train['Access'] == 'all-access') & (train['Agent'] == 'all-agents'))
    idx = idx[0][0]
    
    data = train.iloc[idx,0:-4]
    handle = ax.plot(days, data.values,label=country)
    handles.append(handle)

    ax.legend()
time = np.linspace(0,10,1000)
series = time
series = series + np.random.randn(1000)*0.2
plt.subplots(figsize=(12,6))
plt.plot([i for i in range(1000)],series)
plt.show()
import statsmodels.api as sm
mdl = sm.OLS(time,series).fit()
trend = mdl.predict(time)
plt.subplots(figsize=(12,6))
plt.plot([i for i in range(1000)],series-trend)
plt.show()
## FFT process
from scipy.fftpack import fft
data = train.iloc[:,0:-4]
fft_complex = fft(data)
fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
arr = np.array(fft_mag)
fft_mean = np.mean(arr,axis=0)
fft_xvals = [day/fft_mean.shape[0] for day in range(fft_mean.shape[0])]
npts = len(fft_xvals) // 2+1
fft_mean = fft_mean[:npts]
fft_xvals = fft_xvals[:npts]
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(fft_xvals[1:], fft_mean[1:])
plt.axvline(x=1./7, color='red',alpha=0.3)
plt.axvline(x=2./7, color='red',alpha=0.3)
plt.axvline(x=3./7, color='red',alpha=0.3)
# Forecasting with Neural Networks
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
weekdays = [datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%a') 
            for date in train.columns.values[:-4]]
print(weekdays[0:8])
le = LabelEncoder()
#oe = OrdinalEncoder()
le.fit(weekdays)
day_label = le.fit_transform(weekdays)
# Use LabelEncoder won't encode folllow order
print(day_label[0:8])
print(le.classes_)
day_label = day_label.reshape(-1,1)
print(day_label[0:8])
print(weekdays[0:8])
day_one_hot = OneHotEncoder(sparse=False).fit_transform(day_label)
print(day_one_hot[0:8])
day_one_hot = np.expand_dims(day_one_hot,0)
print(day_one_hot.shape)
agent_int = LabelEncoder().fit(train['Agent'])
agent_enc = agent_int.transform(train['Agent'])
agent_enc = agent_enc.reshape(-1,1)
agent_one_hot = OneHotEncoder(sparse=False).fit(agent_enc)
del agent_enc
page_int = LabelEncoder().fit(train['Sub_Page'])
page_enc = page_int.transform(train['Sub_Page'])
page_enc = page_enc.reshape(-1, 1)
page_one_hot = OneHotEncoder(sparse=False).fit(page_enc)
del page_enc
acc_int = LabelEncoder().fit(train['Access'])
acc_enc = acc_int.transform(train['Access'])
acc_enc = acc_enc.reshape(-1, 1)
acc_one_hot = OneHotEncoder(sparse=False).fit(acc_enc)
del acc_enc
def lag_arr(arr, lag, fill):
    filler = np.full((arr.shape[0],lag,1),-1)
    comb = np.concatenate((filler,arr),axis=1)
    result = comb[:, :arr.shape[1]]
    return result