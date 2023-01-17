# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
%matplotlib inline
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
train = pd.read_csv('../input/train_1.csv').fillna(0)
train.head()
def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]
parse_page(train.Page[0])
l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
df.columns = ['Subject','Sub_Page','Access','Agent']
df.head()
train = pd.concat([train,df],axis=1)
del train['Page']
fig, ax = plt.subplots(figsize=(10, 7))
train.Sub_Page.value_counts().plot(kind='bar')
fig, ax = plt.subplots(figsize=(10, 7))
train.Access.value_counts().plot(kind='bar')
fig, ax = plt.subplots(figsize=(10, 7))
train.Agent.value_counts().plot(kind='bar')
train.head()
from matplotlib import dates

idx = 39457

window = 10


data = train.iloc[idx,0:-4]
name = train.iloc[idx,-4]
days = [r for r in range(data.shape[0] )]

fig, ax = plt.subplots(figsize=(10, 7))

plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title(name)

ax.plot(days,data.values,color='grey')
ax.plot(np.convolve(data, np.ones((window,))/window, mode='valid'),color='black')



ax.set_yscale('log')

fig, ax = plt.subplots(figsize=(10, 7))
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Twenty One Pilots Popularity')
ax.set_yscale('log')
handles = []
for country in ['de','en','es','fr','ru']:
    idx= np.where((train['Subject'] == 'Twenty One Pilots') 
                  & (train['Sub_Page'] == '{}.wikipedia.org'.format(country)) 
                  & (train['Access'] == 'all-access') & (train['Agent'] == 'all-agents'))
    idx=idx[0][0]
    
    data = train.iloc[idx,0:-4]
    handle = ax.plot(days,data.values,label=country)
    handles.append(handle)

ax.legend()
from scipy.fftpack import fft
#idx = 39457
data = train.iloc[:,0:-4]
fft_complex = fft(data)
fft_complex.shape
fft_mag = [np.sqrt(np.real(x)*np.real(x)+
                   np.imag(x)*np.imag(x)) for x in fft_complex]

arr = np.array(fft_mag)
fft_mean = np.mean(arr,axis=0)
fft_mean.shape
fft_xvals = [day / fft_mean.shape[0] for day in range(fft_mean.shape[0])]
npts = len(fft_xvals) // 2 + 1
fft_mean = fft_mean[:npts]
fft_xvals = fft_xvals[:npts]
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fft_xvals[1:],fft_mean[1:])
plt.axvline(x=1./7,color='red',alpha=0.3)
plt.axvline(x=2./7,color='red',alpha=0.3)
plt.axvline(x=3./7,color='red',alpha=0.3)

from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(10, 7))
a = np.random.choice(data.shape[0],1000)

for i in a:
    autocorrelation_plot(data.iloc[i])
    
plt.title('1K Autocorrelations')

fig = plt.figure(figsize=(10, 7))

autocorrelation_plot(data.iloc[110])
plt.title(' '.join(train.loc[110,['Subject', 'Sub_Page']]))
data.shape
from sklearn.model_selection import train_test_split
X = data.iloc[:,:500]
y = data.iloc[:,500:]
X.shape
y.shape
X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, 
                                                  test_size=0.1, 
                                                  random_state=42)
def mape(y_true,y_pred):
    eps = 1
    err = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return err
lookback = 50
lb_data = X_train[:,-lookback:]
med = np.median(lb_data,axis=1,keepdims=True)
eps = 1
err = mape(y_train,med)
err
idx = 15000

fig, ax = plt.subplots(figsize=(10, 7))


ax.plot(np.arange(500),X_train[idx], label='X')
ax.plot(np.arange(500,550),y_train[idx],label='True')

ax.plot(np.arange(500,550),np.repeat(med[idx],50),label='Forecast')

plt.title(' '.join(train.loc[idx,['Subject', 'Sub_Page']]))
ax.legend()
ax.set_yscale('log')
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(X_train[0], order=(5,1,5))
model = model.fit()
model.summary()
fig, ax = plt.subplots(figsize=(10, 7))
idx = 0
residuals = pd.DataFrame(model.resid)
ax.plot(residuals)

plt.title('ARIMA residuals for 2NE1 pageviews')

residuals.plot(kind='kde',
               figsize=(10,7),
               title='ARIMA residual distribution 2NE1 ARIMA', legend=False)
predictions, stderr, conf_int = model.forecast(50)
#target = y_train[0]
fig, ax = plt.subplots(figsize=(10, 7))


ax.plot(np.arange(480,500),X_train[0,480:], label='X')
ax.plot(np.arange(500,550),y_train[0],label='True')

ax.plot(np.arange(500,550),predictions,label='Forecast')


plt.title('2NE1 ARIMA forecasts')
ax.legend()
ax.set_yscale('log')
import simdkalman
smoothing_factor = 5.0

n_seasons = 7

# --- define state transition matrix A
state_transition = np.zeros((n_seasons+1, n_seasons+1))
# hidden level
state_transition[0,0] = 1
# season cycle
state_transition[1,1:-1] = [-1.0] * (n_seasons-1)
state_transition[2:,1:-1] = np.eye(n_seasons-1)
state_transition
observation_model = [[1,1] + [0]*(n_seasons-1)]
observation_model
level_noise = 0.2 / smoothing_factor
observation_noise = 0.2
season_noise = 1e-3

process_noise_cov = np.diag([level_noise, season_noise] + [0]*(n_seasons-1))**2
observation_noise_cov = observation_noise**2
process_noise_cov
observation_noise_cov
kf = simdkalman.KalmanFilter(state_transition = state_transition,
                             process_noise = process_noise_cov,
                             observation_model = observation_model,
                             observation_noise = observation_noise_cov)

result = kf.compute(X_train[0], 50)
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(np.arange(480,500),X_train[0,480:], label='X')
ax.plot(np.arange(500,550),y_train[0],label='True')

ax.plot(np.arange(500,550),
        result.predicted.observations.mean,
        label='Predicted observations')


ax.plot(np.arange(500,550),
        result.predicted.states.mean[:,0],
        label='redicted states')

ax.plot(np.arange(480,500),
        result.smoothed.observations.mean[480:],
        label='Expected Observations')

ax.plot(np.arange(480,500),
        result.smoothed.states.mean[480:,0],
        label='States')



ax.legend()
ax.set_yscale('log')
type(day_one_hot)
day_one_hot = np.expand_dims(day_one_hot,0)
day_one_hot.shape
day_one_hot = np.repeat(day_one_hot,repeats=train.shape[0],axis=0)
def lag_arr(arr, lag,fill):
    filler = np.full((arr.shape[0],lag),-1)
    comb = np.concatenate((filler,arr),axis=1)
    result = comb[:,:log_view.shape[1]]
    return result
year_lag = lag_arr(log_view,365,-1)
year_lag.shape
year_lag = np.expand_dims(year_lag,-1)
halfyear_lag = lag_arr(log_view,182,-1)
halfyear_lag = np.expand_dims(halfyear_lag,-1)
quarter_lag = lag_arr(log_view,91,-1)
quarter_lag = np.expand_dims(quarter_lag,-1)
quarter_lag.shape
agent_enc = LabelEncoder().fit_transform(train['Agent'])
agent_enc = agent_enc.reshape(-1, 1)
agent_enc = OneHotEncoder(sparse=False).fit_transform(agent_enc)
agent_enc.shape
agent_enc = np.expand_dims(agent_enc,1)
agent_enc.shape
log_view.shape
agent_enc = np.repeat(agent_enc,repeats=log_view.shape[1],axis=1)
page_enc = LabelEncoder().fit_transform(train['Sub_Page'])
page_enc = page_enc.reshape(-1, 1)
page_enc = OneHotEncoder(sparse=False).fit_transform(page_enc)
page_enc.shape
page_enc = np.expand_dims(page_enc, 1)
page_enc.shape
acc_enc = LabelEncoder().fit_transform(train['Access'])
acc_enc = acc_enc.reshape(-1, 1)
acc_enc = OneHotEncoder(sparse=False).fit_transform(acc_enc)
acc_enc.shape
acc_enc = np.expand_dims(acc_enc,1)
train.iloc[:5,-4:]
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0
single_autocorr(log_view[0],1)
lag = 365
corrs = []
for i in tqdm(range(train.shape[0])):
    c = single_autocorr(train.iloc[i,:-4].values, lag)
    corrs.append(c)
year_corr = np.array(corrs)
year_corr = year_corr.reshape(-1,1)
year_corr.shape
year_corr = np.expand_dims(year_corr,-1)
lag = 182
corrs = []
for i in tqdm(range(train.shape[0])):
    c = single_autocorr(train.iloc[i,:-4].values, lag)
    corrs.append(c)
halfyear_corr = np.array(corrs)
halfyear_corr = halfyear_corr.reshape(-1,1)
halfyear_corr.shape
halfyear_corr = np.expand_dims(halfyear_corr,-1)
lag = 91
corrs = []
for i in tqdm(range(train.shape[0])):
    c = single_autocorr(train.iloc[i,:-4].values, lag)
    corrs.append(c)
quarter_corr = np.array(corrs)
quarter_corr = quarter_corr.reshape(-1,1)
quarter_corr.shape
quarter_corr = np.expand_dims(quarter_corr,-1)
