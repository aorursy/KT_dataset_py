
#Import Libraries etc etc
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy import signal
import re
%matplotlib inline

#Some nice plotting params
mpl.rcParams['figure.figsize'] = (8,5)
mpl.rcParams['lines.linewidth'] = 3
plt.style.use('ggplot')

#Read in the data.  Seems like the dates are the second last column
df = pd.read_csv('../input/tweets.csv', parse_dates = [-2])

def f(x): # I don't really care about the times that the tweets were made
    
    return dt.datetime(x.year,x.month,x.day)

df['time'] = df.time.apply( f)

time_series = pd.DataFrame(df.time.value_counts().sort_index().resample('D').mean().fillna(0))
time_series.columns = ['Tweets']

time_series.plot()






time_series = time_series.ix['2016-01-28':]
time_series.plot()
ts = time_series.values.ravel()
fs = np.fft.fftfreq(np.arange(len(ts)).shape[-1])
x,y = signal.periodogram(ts)
plt.plot(x,y,'o-')

oneov = [1.0/j for j in range(2,8)]
plt.xticks(oneov,range(2,8))
ax = plt.gca()


ax.grid(True)
ax.set_yticklabels([])
ax.set_xlabel('No. of Days')

p = signal.find_peaks_cwt(ts, np.arange(1,4) )


t = np.arange(len(ts))
plt.plot(t,ts)
plt.plot(t[p],ts[p],'o')
r = time_series.ix[p].reset_index().copy()

r.columns = ['date','tweet']

r['weekday'] = r.date.apply(lambda x: x.weekday())

r.weekday.value_counts()

