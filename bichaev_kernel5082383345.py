# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean



from sklearn.metrics import mean_absolute_error



plt.style.use('ggplot')



warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('/kaggle/input/sputnik/train.csv')

data
x_train = data[data['type'] == 'train']

x_test = data[data['type'] == 'test']

x_train.drop('type', axis=1, inplace=True)

x_test.drop('type', axis=1, inplace=True)
x_train.epoch = pd.to_datetime(x_train.epoch, format='%Y-%m-%d %H:%M:%S.%f')

x_train.index = x_train.epoch

x_train.drop('epoch', axis=1, inplace=True)



x_test.epoch = pd.to_datetime(x_test.epoch, format='%Y-%m-%d %H:%M:%S.%f')

x_test.index = x_test.epoch

x_test.drop('epoch', axis=1, inplace=True)
x_train['error']  = np.linalg.norm(x_train[['x', 'y', 'z']].values - x_train[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
x_one = x_train[x_train.sat_id == 490]

new_data = x_one
new_data.error.plot(figsize=(14,8))
rcParams['figure.figsize'] = 12, 7

result = sm.tsa.seasonal_decompose(x_one.error, model='additive', freq=24)

result.plot()

plt.show()
train = new_data.error[:-24*10]

val = new_data.error[-24*10:]
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



fit1 = ExponentialSmoothing(np.asarray(train) ,seasonal_periods=24, seasonal='add').fit()

forecast = pd.Series(fit1.forecast(len(val)))

forecast.index = val.index
plt.plot(val)

plt.plot(forecast)
# from fbprophet import Prophet

# import copy



# x_one = x_train[x_train.sat_id == 490]



# x_train_one = x_one[:-48]

# val_one = x_one[-48:]



# res = np.array([])



# num_of_obs_daily = x_train_one[x_train_one.index.day == 2].shape[0] # число наблюдений за день

# half_empty_obs = 23 - num_of_obs_daily # считаем, что в трейне всего 23 дня (в 24-м мало часов)

# empty_obs = abs(half_empty_obs) # потерялось(приобрелось) столько часов с учетом сдвигов времени



# loss_during_train = 24*empty_obs # на столько часов сдвинуто на трейне

# nec_pred = val_one.shape[0] // 24 # столько дней нужно предсказать

# loss_during_test = nec_pred*empty_obs # столько дней потеряется на предсказании

# full_loss = loss_during_train + loss_during_test # столько нужно предсказат лишнего и потом на это число сдвинуть





# # fit = ExponentialSmoothing(np.asarray(x_train_one.x), 

# #                            seasonal_periods=24, 

# #                            seasonal='additive').fit()

# # forecast = pd.Series(fit.forecast(val_one.shape[0]+23))





# prophet_data = pd.DataFrame()

# prophet_data['ds'] = x_train_one.index

# prophet_data['y']  = x_train_one.error.values

# prophet_basic = Prophet()

# prophet_basic.fit(prophet_data)

# future = prophet_basic.make_future_dataframe(periods= 48, freq = 'H')

# forecast = prophet_basic.predict(future)

# res = forecast['yhat'].iloc[-48:]





# # if half_empty_obs >=0:

# #     res = np.concatenate((res, forecast.values[23:]))

# # else:

# #     full_loss -= 1

# #     res = np.concatenate((res, forecast.values[23:]))



# val_one['pred'] = res.values
# plt.plot(val_one.error)

# plt.plot(val_one.pred)
res = np.array([])

for i in range(600):

    x_train_one = x_train[x_train.sat_id == i]

    x_test_one = x_test[x_test.sat_id == i]

    

    fit = ExponentialSmoothing(np.asarray(x_train_one.error), 

                               seasonal_periods=24, 

                               seasonal='additive').fit()

    forecast = pd.Series(fit.forecast(x_test_one.shape[0]))

    

    res = np.concatenate((res, forecast.values))

        

x_test['error'] = res
result = x_test[['id', 'error']]

result = result.reset_index().drop('epoch', axis=1)

result.to_csv('sub.csv', index=False)
# res = np.array([])

# for i in range(600):

#     x_train_one = x_train[x_train.sat_id == i]

#     x_test_one = x_test[x_test.sat_id == i]

    

#     fit = ExponentialSmoothing(np.asarray(x_train_one.x), 

#                                seasonal_periods=24, 

#                                seasonal='additive').fit()

#     forecast = pd.Series(fit.forecast(x_test_one.shape[0] + 23))

    

#     res = np.concatenate((res, forecast.values[23:]))

        

# x_test.x = res
# res = np.array([])

# for i in range(600):

#     x_train_one = x_train[x_train.sat_id == i]

#     x_test_one = x_test[x_test.sat_id == i]

    

#     fit = ExponentialSmoothing(np.asarray(x_train_one.y), 

#                                seasonal_periods=24, 

#                                seasonal='additive').fit()

#     forecast = pd.Series(fit.forecast(x_test_one.shape[0] + 23))

    

#     res = np.concatenate((res, forecast.values[23:]))

        

# x_test.y = res
# res = np.array([])

# for i in range(600):

#     x_train_one = x_train[x_train.sat_id == i]

#     x_test_one = x_test[x_test.sat_id == i]

    

#     fit = ExponentialSmoothing(np.asarray(x_train_one.z), 

#                                seasonal_periods=24, 

#                                seasonal='additive').fit()

#     forecast = pd.Series(fit.forecast(x_test_one.shape[0] + 23))



#     res = np.concatenate((res, forecast.values[23:]))

        

# x_test.z = res
# x_test['error']  = np.linalg.norm(x_test[['x', 'y', 'z']].values - x_test[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
# result = x_test[['id', 'error']]

# result = result.reset_index().drop('epoch', axis=1)

# result.to_csv('sub.csv', index=False)
# from statsmodels.tsa.stattools import adfuller

# def test_stationarity(timeseries, window = 24, cutoff = 0.05):



#     #Determing rolling statistics

#     rolmean = timeseries.rolling(window).mean()

#     rolstd = timeseries.rolling(window).std()



#     #Plot rolling statistics:

#     fig = plt.figure(figsize=(12, 4))

#     orig = plt.plot(timeseries, color='blue',label='Original')

#     mean = plt.plot(rolmean, color='red', label='Rolling Mean')

#     plt.legend(loc='best')

#     plt.title('Rolling Mean & Standard Deviation')

#     plt.show()

#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    

#     plt.show()

    

#     #Perform Dickey-Fuller test:

#     print('Results of Dickey-Fuller Test:')

#     dftest = adfuller(timeseries.values,autolag='AIC' )

#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

#     for key,value in dftest[4].items():

#         dfoutput['Critical Value (%s)'%key] = value

#     pvalue = dftest[1]

#     if pvalue < cutoff:

#         print('p-value = %.4f. The series is likely stationary.' % pvalue)

#     else:

#         print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    

#     print(dfoutput)
# x_one.error.plot()
# rcParams['figure.figsize'] = 12, 7

# # x_one_x_diff = x_one.x - x_one.x.shift(24)

# x_one_x_diff = x_one.x - x_one.x.shift(1)

# x_one_x_diff.dropna(inplace = True)

# test_stationarity(x_one_x_diff, window = 24)
# import statsmodels.api as sm

# fig, ax = plt.subplots(figsize=(20,8))

# sm.graphics.tsa.plot_acf(x_one_x_diff.values, lags=50, ax = ax)

# plt.show()
# from itertools import product

# d = 1

# s = 24

# ps = range(0, 3)

# d=1

# qs = range(0, 3)

# Ps = range(0, 3)

# D=1

# Qs = range(0, 3)

# parameters = product(ps, qs, Ps, Qs)

# parameters_list = list(parameters)

# len(parameters_list)
# %%time

# import warnings

# results = []

# best_aic = float("inf")

# warnings.filterwarnings('ignore')



# for param in parameters_list:

#     #try except нужен, потому что на некоторых наборах параметров модель не обучается

#     try:

#         model=sm.tsa.statespace.SARIMAX(x_one.x, order=(param[0], d, param[1]), 

#                                         seasonal_order=(param[2], D, param[3], 24)).fit(disp=-1)

#     #выводим параметры, на которых модель не обучается и переходим к следующему набору

#     except ValueError:

#         print('wrong parameters:', param)

#         continue

#     aic = model.aic

#     #сохраняем лучшую модель, aic, параметры

#     if aic < best_aic:

#         best_model = model

#         best_aic = aic

#         best_param = param

#     results.append([param, model.aic])

    

# warnings.filterwarnings('default')
# # Остатки модели

# plt.figure(figsize=(16,8))

# plt.plot(best_model.resid[1:])

# plt.show()
# stat_test = sm.tsa.adfuller(best_model.resid[:])

# print ('adf: ', stat_test[0] )

# print ('p-value: ', stat_test[1])

# print('Critical values: ', stat_test[4])

# if stat_test[0]> stat_test[4]['5%']: 

#     print ('есть единичные корни, ряд не стационарен')

# else:

#     print ('единичных корней нет, ряд стационарен')
# forecast = best_model.predict(start = x_one.shape[0], end = x_one.shape[0]+73)
# x_one
# x_one_test = x_test[x_test.sat_id == 548]

# x_one_test

# forecast.index = x_one_test.index



# plt.figure(figsize=(16,8))

# # plt.plot(x_one.x, label='Train')

# plt.plot(x_one_test.x_sim, label='Validation')

# plt.plot(forecast, label='SARIMA')

# plt.legend(loc='best')

# plt.show()
# plt.figure(figsize=(16,8))

# # plt.plot(x_one.x, label='Train')

# plt.plot(x_one.x_sim, label='Validation')

# plt.plot(x_one.x, label='SARIMA')

# plt.legend(loc='best')

# plt.show()
# full = pd.concat([x_one[['x','x_sim']], x_one_test[['x','x_sim']]])
# plt.figure(figsize=(16,8))

# # plt.plot(x_one.x, label='Train')

# plt.plot(full.x_sim, label='Validation')

# plt.plot(full.x, label='SARIMA')

# plt.legend(loc='best')

# plt.show()
# from itertools import product

# from multiprocessing import Pool, Lock, Value



# res = []



# def work_with_i(i):

#     print(i)

#     x_train_one = x_train[x_train.sat_id == i].error

#     x_test_one = x_test[x_test.sat_id == i]

#     x_train_one_diff = x_train_one - x_train_one.shift(1)

#     x_train_one_diff.dropna(inplace = True)

#     d = 1

#     s = 24

#     ps = range(0, 2)

#     d=1

#     qs = range(0, 2)

#     Ps = range(0, 2)

#     D=1

#     Qs = range(0, 2)

#     parameters = product(ps, qs, Ps, Qs)

#     parameters_list = list(parameters)



#     results = []

#     best_aic = float("inf")

#     warnings.filterwarnings('ignore')



#     for param in parameters_list:

#         #try except нужен, потому что на некоторых наборах параметров модель не обучается

#         try:

#             model=sm.tsa.statespace.SARIMAX(x_train_one, order=(param[0], d, param[1]), 

#                                             seasonal_order=(param[2], D, param[3], 24),enforce_stationarity=False).fit(disp=-1)

#         #выводим параметры, на которых модель не обучается и переходим к следующему набору

#         except ValueError:

#             print('wrong parameters:', param)

#             continue

#         aic = model.aic

#         #сохраняем лучшую модель, aic, параметры

#         if aic < best_aic:

#             best_model = model

#             best_aic = aic

#             best_param = param

#         results.append([param, model.aic])



#     forecast = best_model.predict(start = x_train_one.shape[0], end = x_train_one.shape[0]+x_test_one.shape[0]-1)

    

#     return i, forecast.values

# #     res = np.concatenate((res, forecast.values))

    

# with Pool(processes=10) as pool:

#     res = pool.map(work_with_i, [i for i in range(600)])

# pool.join()





# # x_test['error'] = res
# with open('result.txt', 'w') as f:

#     for pair in res:

#         f.write(str(pair[0]) + ' ')

#         for num in pair[1]:

#             f.write(str(num) + ' ')

#         f.write('\n')

# # res[0]
# sort_res = sorted(res, key=lambda pair: pair[0])
# result = np.array([])

# for pair in sort_res:

#     result = np.hstack((result, pair[1]))

# result.shape
# x_test['error'] = result

# x_test
# r = x_test[['id', 'error']]

# r = r.reset_index().drop('epoch', axis=1)

# r.to_csv('sub.csv', index=False)