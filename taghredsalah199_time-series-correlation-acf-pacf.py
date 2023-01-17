import numpy as np 

import pandas as pd



df1=pd.read_csv('../input/international-airline-passengers/international-airline-passengers.csv', index_col='Month', parse_dates=True)

df1.index_freq= 'MS'

df2=pd.read_csv('../input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv',index_col='date', parse_dates=True)

df2.index_freq= 'D'



df1.plot() # Has a seasonal and trend component "non_stationary"

df2.plot() # Has no trend o
from pandas.plotting import lag_plot

lag_plot(df1['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60']) #Non stationary cause it has a strong correlation between 2 axis

lag_plot(df2['births']) #Stationary _no correlation_

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df1,lags=40);
plot_acf(df2,lags=40); #Sharp drop-off
plot_pacf(df2,lags=40,title='daily female births'); #pacf work will with stationary data