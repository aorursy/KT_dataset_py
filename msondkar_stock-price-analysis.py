import numpy as np 

import pandas as pd

from pandas import datetime

import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARMA



from sklearn.metrics import mean_squared_error
def parser(x):

    return datetime.strptime(x, '%Y-%m-%d')



# Load the dataset

df = pd.read_csv("../input/amazon-stocks-lifetime-dataset/AMZN.csv", index_col=0, parse_dates=[0], date_parser=parser)
# Show first 5 rows

df.head()
# Statistic summary of the dataset

df.describe()
# Keep only 'Close' column

amzn = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

amzn = amzn[amzn.index >= '2015-01-01']
plt.figure(figsize=(14, 5))

plt.title('Amazon stock closing prices for last 5 years', fontsize=14)

plt.plot(amzn.Close)
Q1_2019_mean = amzn[(amzn.index >= '2019-01-01') & (amzn.index < '2019-03-31')].mean()

Q1_2019_var  = amzn[(amzn.index >= '2019-01-01') & (amzn.index < '2019-03-31')].var()

Q1_2017_mean = amzn[(amzn.index >= '2017-01-01') & (amzn.index < '2017-03-31')].mean()

Q1_2017_var  = amzn[(amzn.index >= '2017-01-01') & (amzn.index < '2017-03-31')].var()

Q4_2015_mean = amzn[(amzn.index >= '2015-10-01') & (amzn.index < '2015-12-31')].mean()

Q4_2015_var  = amzn[(amzn.index >= '2015-10-01') & (amzn.index < '2015-12-31')].var()



print('2019 Quarter 1 closing price mean     : %.2f ' % (Q1_2019_mean))

print('2019 Quarter 1 closing price variance : %.2f ' % (Q1_2019_var))

print("---------------------------------------------- ")

print('2017 Quarter 1 closing price mean     : %.2f ' % (Q1_2017_mean))

print('2017 Quarter 1 closing price variance : %.2f ' % (Q1_2017_var))

print("---------------------------------------------- ")

print('2015 Quarter 4 closing price mean     : %.2f ' % (Q4_2015_mean))

print('2015 Quarter 4 closing price variance : %.2f ' % (Q4_2015_var))
plot_acf(amzn, lags=40)

plt.show()
# Calculate the differnce of a element compared with a prevous row element

amzn_diff = amzn.diff(periods=1)

# Drop rows with NAN value. First row element will have NAN value because there is no previous element for calculating the differnce.

amzn_diff = amzn_diff.dropna()

# Display first five rows

amzn_diff.head()
plt.figure(figsize=(14,5))

plt.title("Amazon closing prices with differencing/integrated order of 1",fontsize=14)

plt.plot(amzn_diff)
plot_acf(amzn_diff, lags=40)

plt.show()
X = amzn.values

size = int(len(X) * 0.70)  # 70 % 

# Training set

train = X[:size]

# testing set

test  = X[size:len(X)]



print("Total Samples    : %d" % len(X))

print("Training Samples : %d" % len(train))

print("Testing Samples  : %d" % len(test))
# train autoregression

ar_model = AR(train)

ar_model_fit = ar_model.fit()

print("Lags : %s" % ar_model_fit.k_ar)

print("Coefficients : %s" % ar_model_fit.params)
# make predictions

preds = ar_model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)



print("AR MModel Mean Squared Error : %.2f" % mean_squared_error(test, preds))

plt.figure(figsize=(14,5))

plt.title("Autoregression Model",fontsize=14)

plt.plot(test)

plt.plot(preds, color='r')
stock_history = [x for x in train ]

preds = []



# order=(p,d,q)

p = 5   # AR parameters/Lags

q = 1    # Differencing order

d = 0    # MA parameters



#arima_model = ARIMA(train, order=(5,1,0))

#arima_model_fit = arima_model.fit()



for i in range(len(test)):

    # initiate ARIMA model

    arima_model = ARIMA(stock_history, order=(p,q,d))

    # fit ARIMA mode;

    arima_model_fit = arima_model.fit()

    # forecast price

    output = arima_model_fit.forecast()[0]

    # append the test price to a stock history data

    stock_history.append(test[i])  

    # append the forcasted price to a list

    preds.append(output)
print("Mean Squared Error : %.2f" % mean_squared_error(test, preds))

plt.figure(figsize=(14,5))

plt.title("Autoregressive Integrated Moving Average Model",fontsize=14)

plt.plot(test, label='Actual Stock Price')

plt.plot(preds, color='r', label='Predicted Stock Price')

plt.legend()
train_test = np.concatenate((train, test))

stock_history = [x for x in train_test]

preds = []

forecasting_days = 300



for i in range(forecasting_days):

    # initiate ARIMA model

    arima_model = ARIMA(stock_history, order=(p,q,d))

    # fit ARIMA mode;

    arima_model_fit = arima_model.fit()

    # forecast price

    output = arima_model_fit.forecast()[0]

    # append the forecasted price to a stock history data

    stock_history.append(output)  

    # append the forcasted price to a prediction list

    preds.append(output)
start = len(train_test)

end = len(train_test) + int(forecasting_days)

history = pd.Series(stock_history)



plt.figure(figsize=(14,5))

plt.title("ARIMA Forecasting for Next 300 Days",fontsize=14)

plt.plot(history[0:start], label='Actual Stock Price')

plt.plot(history[start:end], color='r', label='Forecasted Stock Price')

plt.legend()