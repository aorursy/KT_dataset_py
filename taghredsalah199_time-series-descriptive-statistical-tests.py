import numpy as np 

import pandas as pd



df1=pd.read_csv('../input/international-airline-passengers/international-airline-passengers.csv', index_col='Month', parse_dates=True)

df1.index_freq= 'MS'

df2=pd.read_csv('../input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv',index_col='date', parse_dates=True)

df2.index_freq= 'D'



from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):

    #Pass time series and optimal title, return an ADF report

    print(f'Augmented Dickey-Fuller Test : {title}')

    result = adfuller(series.dropna(),autolag='AIC')#drop nan values

    labels =['ADF Test statistic','P-value','# lags used', '#observations']

    out= pd.Series(result[0:4],index=labels)

    

    for key,val in result[4].items():

        out[f'critical value({key})']=val

    print(out.to_string())

    if result[1]<= 0.05:

        print('Strong evidence against the null hypothesis')

        print('Reject the null hypothesis')

        print('Data has no unit root & is stationary')

    else:

        print('Week evidence against the null hypothesis')

        print('Fail to reject the null hypothesis')

        print('Data has a unit root & is non-stationary')
adf_test(df1['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'])
df1.plot()

# Non-stationary data has a trend and seasonal
adf_test(df2['births'])
df2.plot()

# Stationary data has no trend or seasonal component