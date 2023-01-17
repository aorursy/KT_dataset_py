import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
%matplotlib inline
# read training data
# Assume data is Kepler long-cadence data at 30 minute increments
df_train = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
df_train.head()
amp = df_train.iloc[0,1:]
time = np.arange(0,len(amp))*(30/60)
fig, ax = plt.subplots(1,figsize=(15,5))
fig.suptitle('Sample time history with planet, index 0')
ax.plot(time,amp)
ax.set(xlabel='time(hrs)', ylabel='flux')
ax.grid()
amp_s = scipy.signal.savgol_filter(amp, 101, 2) # window size 51, polynomial order 3
fig, ax = plt.subplots(1,figsize=(15,5))
fig.suptitle('Sample time history with planet, index 0, Smoothed')
ax.plot(time,amp)
ax.plot(time,amp_s,'r-',lw=5)
ax.set(xlabel='time(hrs)', ylabel='flux')
ax.grid()
amp_dt = amp - amp_s
fig, ax = plt.subplots(1,figsize=(15,5))
fig.suptitle('Sample time history with planet, index 0, Detrended')
ax.plot(time,amp_dt)
ax.set(xlabel='time(hrs)', ylabel='flux')
ax.grid()
# subtract mean
amp_m = amp_dt - amp_dt.mean()
fig, ax = plt.subplots(1,figsize=(15,5))
fig.suptitle('Sample time history with planet, index 0, Remove mean bias')
ax.plot(time,amp_m)
ax.set(xlabel='time(hrs)', ylabel='flux')
ax.grid()
# normalize by standard deviation
amp_norm = amp_m / amp_m.std()
fig, ax = plt.subplots(1,figsize=(15,5))
fig.suptitle('Sample time history with planet, index 0, Normalize')
ax.plot(time,amp_norm)
ax.set(xlabel='time(hrs)', ylabel='flux')
ax.grid()
sps = 1 / (30 *60) # samples per second
nt = len(time) # total points in record
nr = 128  # number points per analysis window
df = sps / (nr -1) # delta frequency in hertz
[freq, psd] = signal.welch(amp_norm,fs=sps,window='hann',nperseg=nr)
fig, ax = plt.subplots(1,figsize=(15,5))
fig.suptitle('Sample power spectral density with planet, index 0')
ax.plot(freq,psd)
ax.set(xlabel='Frequency(Hz)', ylabel='flux power')
ax.grid()
# normalize data function
def normalize_data(amp):
    # window size 101, polynomial order 2
    amp_s = scipy.signal.savgol_filter(amp, 101, 2)
    return (amp - amp_s - amp_dt.mean()) / amp_m.std()

# return just the psd value for pandas apply method
def psd_calculation(y,sps,window,nr):
    f,p = signal.welch(y,sps,window,nr)
    return p /  (p.max())
# dataframe with just flux
df_train_flux = df_train.iloc[:,1:]
# normulize
df_train_flux_norm = df_train_flux.apply(normalize_data,axis=1,result_type='broadcast')
df_train_flux_psd = df_train_flux_norm.apply(psd_calculation,axis=1,sps=sps,window='hann',nr=nr,result_type='expand')
df_train_flux_psd.head()
fig, (ax1,ax2) = plt.subplots(2,figsize=(15,5))
fig.suptitle('Time history with planet, index 0, Normalize and psd')

ax1.plot(time,df_train_flux_norm.iloc[0,:])
ax1.set(xlabel='time(hrs)', ylabel='flux')
ax1.grid()

ax2.plot(freq,df_train_flux_psd.iloc[0,:])
ax2.set(xlabel='Frequency(Hz)', ylabel='flux power')
ax2.grid()
df_test = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
df_test.head()
# dataframe with just flux
df_test_flux = df_test.iloc[:,1:]
# normulize
df_test_flux_norm = df_test_flux.apply(normalize_data,axis=1,result_type='broadcast')
df_test_flux_psd = df_test_flux_norm.apply(psd_calculation,axis=1,sps=sps,window='hann',nr=nr,result_type='expand')
df_test_flux_psd.head()
fig, (ax1,ax2) = plt.subplots(2,figsize=(15,5))
fig.suptitle('Time history with planet, index 0, Normalize and psd')

ax1.plot(time,df_test_flux_norm.iloc[0,:])
ax1.set(xlabel='time(hrs)', ylabel='flux')
ax1.grid()

ax2.plot(freq,df_test_flux_psd.iloc[0,:])
ax2.set(xlabel='Frequency(Hz)', ylabel='flux power')
ax2.grid()
# training and test data
X_train = df_train_flux_psd
y_train = df_train['LABEL']
X_test = df_test_flux_psd
y_test = df_test['LABEL']
weights = {1:1.0, 2:5.0}
svcf = SVC(kernel='linear',C=1,class_weight=weights).fit(X_train,y_train)
y_train_pred = svcf.predict(X_train)
y_test_pred = svcf.predict(X_test)
print(confusion_matrix(y_train, y_train_pred, labels=[1, 2]))
print(classification_report(y_train, y_train_pred, labels=[1,2]))
print(confusion_matrix(y_test, y_test_pred, labels=[1, 2]))
print(classification_report(y_test, y_test_pred, labels=[1,2]))
sps = 1 / (30 *60) # samples per second
nt = len(time) # total points in record
nr = 256  # number points per analysis window
df = sps / (nr -1) # delta frequency in hertz
freq = np.arange(0,df*(nr/2+1),df)
#[freq, psd] = signal.welch(amp_norm,fs=sps,window='hann',nperseg=nr)
# normalize data function
def normalize_data(amp):
    amp_dt = amp - amp.shift()
    amp_n = (amp - amp.shift()) / amp_dt.std()
    amp_n[0] = amp_n[1]
    return amp_n
# dataframe with just flux
df_train_flux = df_train.iloc[:,1:]
# normulize
df_train_flux_norm = df_train_flux.apply(normalize_data,axis=1,result_type='broadcast')
df_train_flux_psd = df_train_flux_norm.apply(psd_calculation,axis=1,sps=sps,window='hann',nr=nr,result_type='expand')
df_train_flux_psd.head()
fig, (ax1,ax2) = plt.subplots(2,figsize=(15,5))
fig.suptitle('Time history with planet, index 0, Normalize and psd')

ax1.plot(time,df_train_flux_norm.iloc[0,:])
ax1.set(xlabel='time(hrs)', ylabel='flux')
ax1.grid()

ax2.plot(freq,df_train_flux_psd.iloc[0,:])
ax2.set(xlabel='Frequency(Hz)', ylabel='flux power')
ax2.grid()
#df_test = pd.read_csv('exoTest.csv')
df_test.head()
# dataframe with just flux
df_test_flux = df_test.iloc[:,1:]
# normulize
df_test_flux_norm = df_test_flux.apply(normalize_data,axis=1,result_type='broadcast')
df_test_flux_psd = df_test_flux_norm.apply(psd_calculation,axis=1,sps=sps,window='hann',nr=nr,result_type='expand')
df_test_flux_psd.head()
fig, (ax1,ax2) = plt.subplots(2,figsize=(15,5))
fig.suptitle('Time history with planet, index 0, Normalize and psd')

ax1.plot(time,df_test_flux_norm.iloc[0,:])
ax1.set(xlabel='time(hrs)', ylabel='flux')
ax1.grid()

ax2.plot(freq,df_test_flux_psd.iloc[0,:])
ax2.set(xlabel='Frequency(Hz)', ylabel='flux power')
ax2.grid()
# training and test data
X_train = df_train_flux_psd
y_train = df_train['LABEL']
X_test = df_test_flux_psd
y_test = df_test['LABEL']
weights = {1:1.0, 2:5.0}
svcf = SVC(kernel='linear',C=1,class_weight=weights).fit(X_train,y_train)
y_train_pred = svcf.predict(X_train)
y_test_pred = svcf.predict(X_test)
print(confusion_matrix(y_train, y_train_pred, labels=[1, 2]))
print(classification_report(y_train, y_train_pred, labels=[1,2]))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))