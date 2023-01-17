import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



## timeseries package

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



from datetime import timedelta

from numpy import log
df = pd.read_csv("../input/gold-rate-history-in-tamilnadu-india/gold_rate_history.csv")

df.shape
df.columns
print(f"The dataset available for State: {df['State'].unique()[0]} and City: {df['Location'].unique()[0]}")
## Convert date string to datetime object

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
print(f'The date ranging from {df["Date"].min()} to {df["Date"].max()}')
df["Pure Gold (24 k)"] = df["Pure Gold (24 k)"].astype(np.float)

df["Standard Gold (22 K)"] = df["Standard Gold (22 K)"].astype(np.float)
df.index = df.Date
plt.figure(figsize=(15, 7))

plt.plot(df['Standard Gold (22 K)'])

plt.title('Standard Gold Price (22K) - (Daily data)')

plt.grid(True)

plt.show()
result = adfuller(df['Standard Gold (22 K)'])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])


# Original Series

fig, axes = plt.subplots(3, 2, sharex=False)



fig.set_size_inches(14,13)



axes[0, 0].plot(df['Standard Gold (22 K)'].values); axes[0, 0].set_title('Original Series')

plot_acf(df['Standard Gold (22 K)'].values, ax=axes[0, 1])





# 1st Differencing

axes[1, 0].plot(df['Standard Gold (22 K)'].diff()); axes[1, 0].set_title('1st Order Differencing')

plot_acf(df['Standard Gold (22 K)'].diff().dropna(), ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(df['Standard Gold (22 K)'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')

plot_acf(df['Standard Gold (22 K)'].diff().diff().dropna(), ax=axes[2, 1])



plt.show()
# PACF plot of 1st differenced series

plt.rcParams.update({'figure.figsize':(14,8), 'figure.dpi':150})



## first order differencing

fig, axes = plt.subplots(2, 2, sharex=False)

axes[0, 0].plot(df['Standard Gold (22 K)'].diff()); axes[0, 0].set_title('1st Differencing')

plot_pacf(df['Standard Gold (22 K)'].diff().dropna(), ax=axes[0, 1])



## second order differencing

axes[1, 0].plot(df['Standard Gold (22 K)'].diff().diff()); axes[1, 0].set_title('2nd Differencing')

plot_pacf(df['Standard Gold (22 K)'].diff().diff().dropna(), ax=axes[1, 1])



plt.show()
from statsmodels.tsa.arima_model import ARIMA



# 1,1,2 ARIMA Model

model = ARIMA(df['Standard Gold (22 K)'], order=(1,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Create Training and Test

train = df['Standard Gold (22 K)'][:-200]

test = df['Standard Gold (22 K)'][-200:]



print(train.shape, test.shape)
# Build Model

model = ARIMA(train, order=(1, 2, 1))  

fitted = model.fit()  



# Forecast

fc, se, conf = fitted.forecast(test.shape[0], alpha=0.05)  # 95% conf
# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)
# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
NO_OF_LAG_DAYS = 14
def create_lag_feature(df, no_of_days):

    

    for day in range(1, no_of_days+1):

        df[f"lag_{day}"] = df["Standard Gold (22 K)"].shift(day)

        

    return df
df_features = create_lag_feature(df, NO_OF_LAG_DAYS)
lag_features =  [ col for col in df_features.columns if "lag" in col]

print(lag_features)
df_lag_features = df_features[lag_features + ['Standard Gold (22 K)']].dropna()

df_lag_features.columns
NUM_PREDICTION = 500

train_df = df_lag_features[:-NUM_PREDICTION]

test_df = df_lag_features[-NUM_PREDICTION:]



print(train_df.shape, test_df.shape)
import xgboost as xgb

from sklearn import linear_model

from sklearn.neural_network import MLPRegressor

from sklearn import ensemble

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
model = mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=12,

                           max_iter=100, random_state=5,

                           activation="relu") 
X_train_df = train_df[lag_features]

X_test_df = test_df[lag_features]
model.fit(X_train_df, train_df['Standard Gold (22 K)'])
predicted_price = model.predict(X_test_df)

test_df["22K Gold Predicted_Price"] = predicted_price
test_df[["Standard Gold (22 K)", "22K Gold Predicted_Price"]].head()
# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train_df['Standard Gold (22 K)'], label='training')

plt.plot(test_df['Standard Gold (22 K)'], label='actual')

plt.plot(test_df['22K Gold Predicted_Price'], label='forecast')

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()
print(f"RMSE {mean_squared_error(test_df['Standard Gold (22 K)'], test_df['22K Gold Predicted_Price'], squared= False)}")
# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(test_df['Standard Gold (22 K)'][-NUM_PREDICTION:], label='actual')

plt.plot(test_df['22K Gold Predicted_Price'][-NUM_PREDICTION:], label='forecast')

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()