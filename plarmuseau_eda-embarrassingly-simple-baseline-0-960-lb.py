import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import math
df = pd.read_csv("../input/liverpool-ion-switching/test.csv")

df
n_groups = 40

df["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    df.loc[ids,"group"] = i
for i in range(n_groups):

    sub = df[df.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin) 

    df.loc[sub.index,"open_channels"] = [0,] + list(np.array(signals[:-1],np.int))
import matplotlib.pyplot as plt

#distr=df.groupby('open_channels').count() 



plt.scatter(df['signal'],df['open_channels'] ,c=df.group)

plt.show()

from scipy.stats import boxcox

trdf = pd.read_csv("../input/liverpool-ion-switching/train.csv")

n_groups = 100

trdf["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*50000, (i+1)*50000)

    trdf.loc[ids,"group"] = i
for i in range(n_groups):

    sub = trdf[trdf.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin) 

    trdf.loc[sub.index,"open_channel2"] = [0,] + list(np.array(signals[:-1],np.int))
plt.scatter(trdf[-500000:]['open_channel2'],trdf[-500000:]['open_channels'] ,c=trdf[-500000:]['group'])

plt.show()

plt.scatter(trdf[-500000:]['signal'],trdf[-500000:]['open_channels'] ,c=trdf[-500000:]['group'])

plt.show()

trdf
lengte=50000

tr=trdf[:lengte]

tr['date']=pd.to_datetime((tr.time*1000000-1.469100e+06)*100, unit='ms')



ts = pd.Series(tr.signal.values, index=pd.DatetimeIndex(tr.date).to_period('ms'))

ts
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error

from math import sqrt



#Checking trend and autocorrelation

def initial_plots(time_series, num_lag):



    #Original timeseries plot

    plt.figure(1)

    plt.plot(time_series)

    plt.title('Original data across time')

    plt.figure(2)

    plot_acf(time_series, lags = num_lag)

    plt.title('Autocorrelation plot')

    plot_pacf(time_series, lags = num_lag)

    plt.title('Partial autocorrelation plot')

    

    plt.show()



    

#Augmented Dickey-Fuller test for stationarity

#checking p-value

#print('p-value: {}'.format(adfuller('date')[1]))



#plotting

initial_plots(ts, 45)
#Defining RMSE

def rmse(x,y):

    return sqrt(mean_squared_error(x,y))



#fitting ARIMA model on dataset

def SARIMAX_call(time_series,p_list,d_list,q_list,P_list,D_list,Q_list,s_list,test_period):    

    

    #Splitting into training and testing

    training_ts = time_series[:-test_period]

    

    testing_ts = time_series[len(time_series)-test_period:]

    

    error_table = pd.DataFrame(columns = ['p','d','q','P','D','Q','s','AIC','BIC','RMSE'],\

                                                           index = range(len(ns_ar)*len(ns_diff)*len(ns_ma)*len(s_ar)\

                                                                         *len(s_diff)*len(s_ma)*len(s_list)))

    count = 0

    

    for p in p_list:

        for d in d_list:

            for q in q_list:

                for P in P_list:

                    for D in D_list:

                        for Q in Q_list:

                            for s in s_list:

                                #fitting the model

                                SARIMAX_model = SARIMAX(training_ts.astype(float),\

                                                        order=(p,d,q),\

                                                        seasonal_order=(P,D,Q,s),\

                                                        enforce_invertibility=False)

                                SARIMAX_model_fit = SARIMAX_model.fit(disp=0)

                                AIC = np.round(SARIMAX_model_fit.aic,2)

                                BIC = np.round(SARIMAX_model_fit.bic,2)

                                predictions = SARIMAX_model_fit.forecast(steps=test_period,typ='levels')

                                RMSE = rmse(testing_ts.values,predictions.values)                                

                                print(p,d,q,P,D,Q,AIC,BIC,RMSE)

                                #populating error table

                                error_table['p'][count] = p

                                error_table['d'][count] = d

                                error_table['q'][count] = q

                                error_table['P'][count] = P

                                error_table['D'][count] = D

                                error_table['Q'][count] = Q

                                error_table['s'][count] = s

                                error_table['AIC'][count] = AIC

                                error_table['BIC'][count] = BIC

                                error_table['RMSE'][count] = RMSE

                                

                                count+=1 #incrementing count        

    

    #returning the fitted model and values

    return error_table



ns_ar = [0,1,2]

ns_diff = [1]

ns_ma = [0,1]#,2]

s_ar = [0,1]

s_diff = [0,1] 

s_ma = [1,2]

s_list = [4]



error_table = SARIMAX_call(ts,ns_ar,ns_diff,ns_ma,s_ar,s_diff,s_ma,s_list,30)

error_table.sort_values(by='RMSE').head(15)
import statsmodels.api as sm



mod = sm.tsa.SARIMAX(ts,order = (0,1,1), seasonal_order=(0,0,1,4)).fit()

mod.summary()
mod.plot_diagnostics()
sample_df = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
sample_df.open_channels = np.array(df.open_channels, np.int)

sample_df.to_csv("submission.csv",index=False)