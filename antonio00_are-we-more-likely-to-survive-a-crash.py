import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline
AirCrashPd = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv',sep=',')
AirCrashPd.head()
plt.scatter(AirCrashPd['Aboard'],AirCrashPd['Fatalities'],alpha=0.7,s = 50)

plt.xlabel('Aboard')

plt.ylabel('Fatalities')
AirCrashPd['survivors']=100*(AirCrashPd.Aboard-AirCrashPd.Fatalities)/AirCrashPd.Aboard
AirCrashPd['survivors'].dropna().plot(kind='hist')

plt.xlabel('% survivors')
AirCrashPd['Date']=pd.to_datetime(AirCrashPd['Date'])
survivors_series=AirCrashPd.groupby(AirCrashPd['Date'].dt.year)['survivors'].mean()

survivors_series=pd.Series(survivors_series,index=survivors_series.index)
survivors_series.dropna().plot()

plt.ylabel('% survivors')
from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(survivors_series.diff().dropna())