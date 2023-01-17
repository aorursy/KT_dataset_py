# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

inputfile = '../input/avocado-prices/avocado.csv'

outputfile = '../ouput/temp.xml'



# 读取数据

data = pd.read_csv(inputfile)
# 展示前部分数据

data.head()
# 数据信息

data.info()
# 数据基本情况

statistics = data.describe()

print(statistics)
# 数据异常值分析

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False



plt.figure(figsize=(15,8))

data.boxplot()

data.boxplot(figsize=(15,8), rot=90, column='AveragePrice', by=['type','year','region'], return_type='dict')

data.boxplot(figsize=(15,8), rot=90, column='Total Volume', by=['region'], return_type='dict')



plt.show()
# 数据采集

data_c = data[data['type']=='conventional'][['Date', 'AveragePrice', 'Total Volume', 'year']]

data_o = data[data['type']=='organic'][['Date', 'AveragePrice', 'Total Volume', 'year']]

data_a = data[['Date', 'AveragePrice', 'Total Volume', 'year']]
# 对数据进行日期格式化

data_c.Timestamp = pd.to_datetime(data_c.Date, format='%Y-%m-%d')

data_c.index = data_c.Timestamp

data_o.Timestamp = pd.to_datetime(data_o.Date, format='%Y-%m-%d')

data_o.index = data_o.Timestamp

data_a.Timestamp = pd.to_datetime(data_a.Date, format='%Y-%m-%d')

data_a.index = data_a.Timestamp
# 按照不同的时间进行数据重新采集

# avergePC = data_c['AveragePrice'].resample('W').mean()

# avergePO = data_o['AveragePrice'].resample('W').mean()

# avergePA = data_a['AveragePrice'].resample('W').mean()

avergePC = data_c['AveragePrice'].resample('M').mean()

avergePO = data_o['AveragePrice'].resample('M').mean()

avergePA = data_a['AveragePrice'].resample('M').mean()



# 绘制变化趋势图

avergePC.plot(figsize=(15,8), label='常规', title='鳄梨平均价格变化趋势', fontsize=14)

avergePO.plot(figsize=(15,8), label='有机', title='鳄梨平均价格变化趋势', fontsize=14)

avergePA.plot(figsize=(15,8), label='全部', title='鳄梨平均价格变化趋势', fontsize=14)

plt.legend(loc='best')

plt.show()
# Decomposing-分解

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(avergePA, freq=12)

trend = decomposition.trend   # 趋势部分

seasonal = decomposition.seasonal # 季节性部分

residual = decomposition.resid # 残留部分

decomposition.plot()
# 划分训练数据和测试数据

l = int(len(avergePA)*0.8)

train = avergePA[:l]

test = avergePA[l:]
# 自相关图 

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(avergePA).show()



# 平稳性检测

from statsmodels.tsa.stattools import adfuller as ADF

print('原始序列的ADF检测结果：', ADF(avergePA))



# 差分后结果

D_train = avergePA.diff().dropna()

plot_acf(D_train).show() # 自相关图

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(D_train).show() # 偏自相关图

print('差分序列的ADF检验结果为：', ADF(D_train))



# 白噪声检测

from statsmodels.stats.diagnostic import acorr_ljungbox

print('一阶差分后的白噪声检测结果为：', acorr_ljungbox(D_train, lags=1))
from statsmodels.tsa.statespace.sarimax import SARIMAX

import itertools

# 定阶

p = q = range(0, 2)

d = range(1,2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

bic_matrix = [] # bic矩阵

for param in pdq:

    tmp = []

    for param_seasonal in seasonal_pdq:

        try:

            results = SARIMAX(train, 

                              order=param, 

                              seasonal_order=param_seasonal, 

                              enforce_stationarity=False, 

                              enforce_invertibility=False).fit().aic

            tmp.append(results)

            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results))

        except:

            tmp.append(None)

    bic_matrix.append(tmp)



bic_matrix = pd.DataFrame(bic_matrix)

p,q = bic_matrix.stack().astype('float64').idxmin()

print('AIC最小的ARIMA为：', pdq[p], seasonal_pdq[q])



# 建立SARIMAX模型

model = SARIMAX(train, order=pdq[p], 

                seasonal_order=seasonal_pdq[q], 

                enforce_stationarity=False, enforce_invertibility=False).fit()

print(model.summary())

model.plot_diagnostics(figsize=(16,8))



# SARIMAX模型预测

pred = model.predict(start=test.index[0], end=test.index[-1], dynamic=True)

# pred2 = model.forecast(15)

plt.figure(figsize=(16,8))

plt.plot(train, label='Train')

plt.plot(test, label='Test')

plt.plot(pred, label='SARIMA')

# plt.plot(pred2, label='SARIMA2')

plt.legend(loc='best')

plt.show()
# Holt-Winters模型

from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12).fit(use_basinhopping = True, use_boxcox = 'log')

pred = model.predict(start=test.index[0], end=test.index[-1])

# pred = model.forecast(15)



plt.figure(figsize=(16,8))

plt.plot(train.index, train, label='Train')

plt.plot(test.index, test, label='Test')

plt.plot(pred.index, pred, label='Holt-Winters')

plt.legend(loc='best')