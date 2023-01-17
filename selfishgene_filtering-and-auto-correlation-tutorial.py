import numpy as np

import pandas as pd

from pandas.plotting import autocorrelation_plot, lag_plot

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
#%% load data that will be used in the script

cityTable     = pd.read_csv('../input/city_attributes.csv')

temperatureDF = pd.read_csv('../input/temperature.csv', index_col=0)

temperatureDF.index = pd.to_datetime(temperatureDF.index)



cityTable
#%% show several temperature plots to get a feel for the dataset

#citiesToShow = ['San Francisco','Las Vegas','Chicago','Toronto','Houston','Jerusalem']

citiesToShow  = ['Portland','Dallas','Miami','Montreal','Tel Aviv District']



t0 = temperatureDF.index

t1 = pd.date_range(pd.to_datetime('1/7/2015',dayfirst=True),pd.to_datetime('1/10/2016',dayfirst=True),freq='H')

t2 = pd.date_range(pd.to_datetime('1/7/2015',dayfirst=True),pd.to_datetime('1/9/2015' ,dayfirst=True),freq='H')

t3 = pd.date_range(pd.to_datetime('1/7/2015',dayfirst=True),pd.to_datetime('21/7/2015',dayfirst=True),freq='H')

t = [t0, t1, t2, t3]



fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(15,14))

for i, t in enumerate(t):

    for k in range(len(citiesToShow)):

        ax[i].plot(t,temperatureDF.loc[t,citiesToShow[k]])



ax[0].legend(citiesToShow, fontsize=16,

              loc='upper left',bbox_to_anchor=(0.02,1.3), ncol=len(citiesToShow))

for i in range(len(ax)): ax[i].set_ylabel('Temperature [$^\circ$K]', fontsize=11)

ax[3].set_xlabel('time', fontsize=14);

#%% show autocorr and lag plots



cityToShow = 'Los Angeles'

selectedLagPoints = [1,3,6,9,12,24,36,48,60]

maxLagDays = 7



originalSignal = temperatureDF[cityToShow]



# set grid spec of the subplots

plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(2, len(selectedLagPoints))

axTopRow = plt.subplot(gs[0, :])

axBottomRow = []

for i in range(len(selectedLagPoints)):

    axBottomRow.append(plt.subplot(gs[1, i]))



# plot autocorr

allTimeLags = np.arange(1,maxLagDays*24)

autoCorr = [originalSignal.autocorr(lag=dt) for dt in allTimeLags]

axTopRow.plot(allTimeLags,autoCorr); 

axTopRow.set_title('Autocorrelation Plot of Temperature Signal', fontsize=18);

axTopRow.set_xlabel('time lag [hours]'); axTopRow.set_ylabel('correlation coefficient')

selectedAutoCorr = [originalSignal.autocorr(lag=dt) for dt in selectedLagPoints]

axTopRow.scatter(x=selectedLagPoints, y=selectedAutoCorr, s=50, c='r')



# plot scatter plot of selected points

for i in range(len(selectedLagPoints)):

    lag_plot(originalSignal, lag=selectedLagPoints[i], s=0.5, alpha=0.7, ax=axBottomRow[i])    

    if i >= 1:

        axBottomRow[i].set_yticks([],[])

plt.tight_layout()
#%% zoom in and out on the autocorr plot

fig, ax = plt.subplots(nrows=4,ncols=1, figsize=(14,14))



timeLags = np.arange(1,25*24*30)

autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]

ax[0].plot(1.0/(24*30)*timeLags, autoCorr); ax[0].set_title('Autocorrelation Plot', fontsize=20);

ax[0].set_xlabel('time lag [months]'); ax[0].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,20*24*7)

autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]

ax[1].plot(1.0/(24*7)*timeLags, autoCorr);

ax[1].set_xlabel('time lag [weeks]'); ax[1].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,20*24)

autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]

ax[2].plot(1.0/24*timeLags, autoCorr);

ax[2].set_xlabel('time lag [days]'); ax[2].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,3*24)

autoCorr = [originalSignal.autocorr(lag=dt) for dt in timeLags]

ax[3].plot(timeLags, autoCorr);

ax[3].set_xlabel('time lag [hours]'); ax[3].set_ylabel('correlation coeff', fontsize=12);
#%% apply rolling mean and plot the signal (low pass filter)

windowSize = 5*24



lowPassFilteredSignal = originalSignal.rolling(windowSize, center=True).mean()



t0 = temperatureDF.index

t1 = pd.date_range(pd.to_datetime('1/7/2015',dayfirst=True),

                   pd.to_datetime('1/10/2016',dayfirst=True),freq='H')

t2 = pd.date_range(pd.to_datetime('1/7/2015',dayfirst=True),

                   pd.to_datetime('1/9/2015' ,dayfirst=True),freq='H')

t3 = pd.date_range(pd.to_datetime('1/7/2015',dayfirst=True),

                   pd.to_datetime('21/7/2015',dayfirst=True),freq='H')



fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(14,12))

ax[0].plot(t0,originalSignal,c='y')

ax[0].plot(t0,lowPassFilteredSignal,c='r')



ax[1].plot(t1,originalSignal[t1],c='y')

ax[1].plot(t1,lowPassFilteredSignal[t1],c='r')



ax[2].plot(t2,originalSignal[t2],c='y')

ax[2].plot(t2,lowPassFilteredSignal[t2],c='r')



ax[3].plot(t3,originalSignal[t3],c='y')

ax[3].plot(t3,lowPassFilteredSignal[t3],c='r')



ax[0].legend(['original signal','low pass filtered'], fontsize=18,

              loc='upper left',bbox_to_anchor=(0.02,1.4), ncol=len(citiesToShow))

for i in range(len(ax)): ax[i].set_ylabel('Temperature [$^\circ$K]', fontsize=11)

ax[3].set_xlabel('time', fontsize=14);

#%% subtract the low pass filtered singal from the original to get high pass filtered signal

highPassFilteredSignal = originalSignal - lowPassFilteredSignal



fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(14,12))

ax[0].plot(t0,highPassFilteredSignal,c='k')

ax[1].plot(t1,highPassFilteredSignal[t1],c='k')

ax[2].plot(t2,highPassFilteredSignal[t2],c='k')

ax[3].plot(t3,highPassFilteredSignal[t3],c='k')



ax[0].set_title('Deflection of Temperature from local mean',fontsize=20)

for i in range(len(ax)): ax[i].set_ylabel('$\Delta$ Temperature [$^\circ$K]', fontsize=11)

ax[3].set_xlabel('time', fontsize=14);
#%% autocorr of low pass filtered singal

fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(14,14))



timeLags = np.arange(1,25*24*30)

autoCorr = [lowPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[0].plot(1.0/(24*30)*timeLags, autoCorr); 

ax[0].set_title('Autocorrelation Plot of Low Pass Filtered Signal', fontsize=20);

ax[0].set_xlabel('time lag [months]'); ax[0].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,20*24*7)

autoCorr = [lowPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[1].plot(1.0/(24*7)*timeLags, autoCorr);

ax[1].set_xlabel('time lag [weeks]'); ax[1].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,20*24)

autoCorr = [lowPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[2].plot(1.0/24*timeLags, autoCorr);

ax[2].set_xlabel('time lag [days]'); ax[2].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,3*24)

autoCorr = [lowPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[3].plot(timeLags, autoCorr);

ax[3].set_xlabel('time lag [hours]'); ax[3].set_ylabel('correlation coeff', fontsize=12);
#%% autocorr of high pass filtered signal

fig, ax = plt.subplots(nrows=4,ncols=1, figsize=(14,14))



timeLags = np.arange(1,25*24*30)

autoCorr = [highPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[0].plot(1.0/(24*30)*timeLags, autoCorr); 

ax[0].set_title('Autocorrelation Plot of High Pass Filtered Signal', fontsize=20);

ax[0].set_xlabel('time lag [months]'); ax[0].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,20*24*7)

autoCorr = [highPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[1].plot(1.0/(24*7)*timeLags, autoCorr);

ax[1].set_xlabel('time lag [weeks]'); ax[1].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,20*24)

autoCorr = [highPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[2].plot(1.0/24*timeLags, autoCorr);

ax[2].set_xlabel('time lag [days]'); ax[2].set_ylabel('correlation coeff', fontsize=12);



timeLags = np.arange(1,3*24)

autoCorr = [highPassFilteredSignal.autocorr(lag=dt) for dt in timeLags]

ax[3].plot(timeLags, autoCorr);

ax[3].set_xlabel('time lag [hours]'); ax[3].set_ylabel('correlation coeff', fontsize=12);