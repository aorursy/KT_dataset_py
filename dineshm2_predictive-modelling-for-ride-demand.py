import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error

from math import sqrt
df = pd.read_csv("../input/ct_rr.csv", parse_dates=['ts'])
df.shape
df.head()
df.dtypes
df.isnull().sum()
df.describe().T
# Feature Engineering

def create_day_series(df):

    

    # Grouping by Date/Time to calculate number of trips

    day_df = pd.Series(df.groupby(['ts']).size())

    # setting Date/Time as index

    day_df.index = pd.DatetimeIndex(day_df.index)

    # Resampling to daily trips

    day_df = day_df.resample('1D').apply(np.sum)

    

    return day_df



day_df_2018 = create_day_series(df)

day_df_2018.head()
day_df_2018.shape
from random import randrange

from pandas import Series

from matplotlib import pyplot

from statsmodels.tsa.seasonal import seasonal_decompose



result = seasonal_decompose(day_df_2018, model='additive', freq=7)

result.plot()

pyplot.show()
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

print('p-value: {}'.format(adfuller(day_df_2018)[1]))



#plotting

initial_plots(day_df_2018, 45)
#storing differenced series

diff_series = day_df_2018.diff(periods=1)



#Augmented Dickey-Fuller test for stationarity

#checking p-value

print('p-value: {}'.format(adfuller(diff_series.dropna())[1]))
initial_plots(diff_series.dropna(), 30)
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

                                                        enforce_invertibility=False,enforce_stationarity=False)

                                

                                SARIMAX_model_fit = SARIMAX_model.fit(disp=0)

                                AIC = np.round(SARIMAX_model_fit.aic,2)

                                BIC = np.round(SARIMAX_model_fit.bic,2)

                                predictions = SARIMAX_model_fit.forecast(steps=test_period,typ='levels')

                                RMSE = rmse(testing_ts.values,predictions.values)                                



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

ns_ma = [0,1,2]

s_ar = [0,1]

s_diff = [0,1] 

s_ma = [1,2]

s_list = [7]



error_table = SARIMAX_call(day_df_2018,ns_ar,ns_diff,ns_ma,s_ar,s_diff,s_ma,s_list,30)
# printing top 5 lowest RMSE from error table

error_table.sort_values(by='RMSE').head(5)
#Predicting values using the fitted model

def predict(time_series,p,d,q,P,D,Q,s,n_days,conf):

    

    #Splitting into training and testing

    training_ts = time_series[:-n_days]

    

    testing_ts = time_series[len(time_series)-n_days:]

    

    #fitting the model

    SARIMAX_model = SARIMAX(training_ts.astype(float),\

                            order=(p,d,q),\

                            seasonal_order=(P,D,Q,s),\

                            enforce_invertibility=False)

    SARIMAX_model_fit = SARIMAX_model.fit(disp=0)

    

    #Predicting

    SARIMAX_prediction = pd.DataFrame(SARIMAX_model_fit.forecast(steps=n_days,alpha=(1-conf)).values,\

                          columns=['Prediction'])

    SARIMAX_prediction.index = pd.date_range(training_ts.index.max()+1,periods=n_days)

    

    #Plotting

    plt.figure(figsize=(20,10))

    plt.title('Plot of original data and predicted values using the ARIMA model')

    plt.xlabel('Time')

    plt.ylabel('Number of Trips')

    plt.plot(time_series[1:],'k-', label='Original data')

    plt.plot(SARIMAX_prediction,'r--', label='Next {}days predicted values'.format(n_days))

    plt.legend()

    plt.show()

    

    #Returning predicitons

    return SARIMAX_prediction



#Predicting the values and builing an 80% confidence interval

prediction = predict(day_df_2018,0,1,0,0,1,2,7,20,0.80)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train, test = day_df_2018.iloc[0:300], day_df_2018.iloc[300:]

model = ExponentialSmoothing(day_df_2018, seasonal='mul', seasonal_periods=7).fit()

pred = model.predict(start=test.index[0], end=test.index[-1])

plt.figure(figsize=(15,9))

plt.plot(train.index, train, label='Train')

plt.plot(test.index, test, label='Test')

plt.plot(pred.index, pred, label='Holt-Winters')

plt.legend(loc='best')



#plt.gcf().set_size_inches(20, 10)
