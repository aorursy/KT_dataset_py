import pandas as pd
df = pd.read_csv('../input/perrin-freres-monthly-champagne-sales/Perrin Freres monthly champagne sales millions.csv')
df.head()
df.tail()
df.isnull().sum()
df.drop([105,106], axis = 0, inplace = True)
df.isnull().sum()
df.columns = ['Month', 'Sales']
df.head()
df.shape
df.dtypes
df['Month'] = pd.to_datetime(df['Month'])
df.dtypes
df.head()
df.set_index('Month', inplace = True)
df.head()
df.plot()
from statsmodels.tsa.stattools import adfuller

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF test statistics', 'P-value', '#Lags used', 'Number of observation used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis (Ho), Reject the null hypothesis, Data has no unit root and is stationary')
    else:
        print('Weak evidence against the null hypothesis (Ho), time series has a unit root, indicating it is non stationary. ')
        
        
adfuller_test(df['Sales'])
df['seasional_first_difference'] = df['Sales'] - df['Sales'].shift(12)
df
adfuller_test(df['seasional_first_difference'].dropna())
df['seasional_first_difference'].plot()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['seasional_first_difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['seasional_first_difference'].iloc[13:],lags=40,ax=ax2)
from statsmodels.tsa.arima_model import ARIMA

model_arima = ARIMA(df['Sales'], order = (1,1,1))     #order = (p, d, q)
model_arima_fit = model_arima.fit()
model_arima_fit.summary()
df['forecast_arima']=model_arima_fit.predict(start=90, end=103, dynamic=True)
df[['Sales','forecast_arima']].plot()
model_sarimax = sm.tsa.statespace.SARIMAX(df['Sales'],
                                          order = (1,1,1),                  # order = (p, d, q)
                                          seasonal_order = (1,1,1, 12))     # seasonal_order = (p, d, q, shift)  
model_sarimax_fit = model_sarimax.fit()
df['forecast_sarimax']=model_sarimax_fit.predict(start=90, end=103, dynamic=True)      # 90 and 103 are the index range to be predicted
df[['Sales','forecast_sarimax']].plot(figsize = (12, 8))
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset (months = x) for x in range(0, 24)]
future_date_dataset = pd.DataFrame(index = future_dates[1:], columns = df.columns)
future_date_dataset.tail()
merged_df = pd.concat([df, future_date_dataset])
merged_df.tail()
df.tail()
merged_df['forecast_sarimax'] = model_sarimax_fit.predict(start =104, end = 120, dynamic= True )
merged_df[['Sales', 'forecast_sarimax']].plot(figsize = (12, 8))