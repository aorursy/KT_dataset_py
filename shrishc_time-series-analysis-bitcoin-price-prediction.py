import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.arima_model import ARIMA



from scipy.stats import boxcox



from itertools import product

from numpy.linalg import LinAlgError
df = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv')  



#just a few basic steps

print(df.head())

print(df.shape)

print(df.describe())

print(df.isnull().any())
df.Timestamp = pd.to_datetime(df.Timestamp, unit = 's')

df.index = df.Timestamp

df = df.resample('M').mean()



print(df.describe())

df.isnull().any()

prices = df.Weighted_Price
plt.figure(figsize = (14,6))

sns.lineplot(x = df.index, y = prices)
def decompose(series):

    plt.figure(figsize = (14,7))

    seasonal_decompose(series).plot()

    plt.show()

    

def DFTest(series):

    testdf = adfuller(series)

    print("DF test p-value : %.16f" %testdf[1] )

    

    

def plots(series):

    plt.figure(figsize = (10,6))

    sns.lineplot(data = series, color = 'blue', label = 'observed line plot')

    sns.lineplot(data = series.rolling(window = 12).mean(), color = 'green', label = 'rolling mean, window -12')

    sns.lineplot(data = series.rolling(window = 12).std(), color = 'black', label = 'std deviation, window -12')

    
print("DF Test->")

#running tests

DFTest(prices)

decompose(prices)

plots(prices)
prices_log = np.log(prices)



#running tests

DFTest(prices_log)

decompose(prices_log)

plots(prices_log)



#prices_log with regular shift transform

prices_log_r = prices_log - prices_log.shift(1)

prices_log_r.dropna(inplace = True)



DFTest(prices_log_r)

decompose(prices_log_r)

plots(prices_log_r)
prices_box_cox_, lambda_ = boxcox(prices)

prices_box_cox = pd.Series(data = prices_box_cox_, index = df.index) #decompose functions requires a pandas object that has a timestamp index.



decompose(prices_box_cox) 

DFTest(prices_box_cox)

print('lambda value:', lambda_)

plots(prices_box_cox)
prices_box_cox_r = prices_box_cox - prices_box_cox.shift(1)

prices_box_cox_r.dropna(inplace = True)



decompose(prices_box_cox_r) 

DFTest(prices_box_cox_r)

plots(prices_box_cox_r)
plt.figure(figsize = (14,7)) 

a = acf(prices_log_r)

p = pacf(prices_log_r)



plt.subplot(221)

sns.lineplot(data = a)

plt.axhline(y=0, linestyle='--', color='gray')



plt.subplot(222)

sns.lineplot(data = p)

plt.axhline(y=0, linestyle='--', color='gray')
a = [[1,2,3], [1],[1,2,3]]

params = list(product(*a))



results = []   

min_aic = float('inf')

best_param = []



# checking different set of params for best fit

for param in params:

    try:

        model = ARIMA(prices_log, order = param).fit(disp = -1)

    except LinAlgError:

        print('Rejected Parameters:', param)

        continue

    except ValueError:

        print('Rejected Parameters:', param)

        continue

    if(min_aic > model.aic):

        min_aic = model.aic

        best_param = param

        best_model = model

        

    results.append([param, model.aic])



print(best_param,min_aic)

print(results)



print(best_model.fittedvalues)



plt.figure(figsize=(16,8))

sns.lineplot(data = prices_log_r, color = 'blue')

sns.lineplot(data = best_model.fittedvalues, color = 'red')    
fitted_values = best_model.fittedvalues

fitted_values = fitted_values.cumsum()



fitted_values = fitted_values + prices_log[0]



final_values = np.exp(fitted_values)



d = {'prices' : prices, 'prices_log' : prices_log, 'price_log_r' : prices_log_r, 'fitted_values' : fitted_values, 'final_values' : final_values}

summaryDF = pd.DataFrame(data = d)

sns.lineplot(data = summaryDF['prices'], color = 'blue')

sns.lineplot(data = summaryDF['final_values'], color = 'red')
predicted_values = np.exp((best_model.predict(start = 1, end = 99).cumsum()) + prices_log[0])

sns.lineplot(data = prices, label  = 'recorded')

sns.lineplot(data = predicted_values, label = 'predicted')
