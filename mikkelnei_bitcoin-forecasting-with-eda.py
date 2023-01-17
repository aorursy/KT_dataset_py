import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/bitcoin-price-prediction/bitcoin_price_Training - Training.csv')

df
df.info()
df.Date = pd.to_datetime(df["Date"])

df.Date.dtype
print('Date starts: ',df.Date.min())

print('Date ends: ',df.Date.max())
df.describe()
sns.lineplot(df.Date,df.Open,color='#FF7433')
sns.lineplot(df.Date,df.Close,color='#FFC300')
sns.lineplot(df.Date,df.Low,color='#3371FF')
sns.lineplot(df.Date,df.High,color='#FF6833')
y = df[['Date','Open']]
y = y.set_index('Date')
sns.distplot(y)

plt.axvline(y.Open.mean())
# time series



import statsmodels.api as sm

decomposition = sm.tsa.seasonal_decompose(y,model='multiplicative')

decomposition.plot()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



plot_acf(y,lags=30)

plot_pacf(y,lags=30)
# plot rolling means and standard deviation



from statsmodels.tsa.stattools import adfuller



def test_stationarity(y):

  rol_mean = y.rolling(12).mean()

  rol_std = y.rolling(12).std()



  plt.subplots(figsize=(15,8))

  plt.plot(rol_mean,color='red',label='rolling_mean')

  plt.plot(rol_std,color='black',label='rolling_std')

  plt.plot(y,color='blue',label='original')

  plt.legend(loc='best')

  plt.show()



  df_test = adfuller(y.Open,autolag='AIC')

  df_output = pd.Series(df_test[0:4],index=['test stats','p-value','#lags','no. of oservations used'])

  print(df_output)
test_stationarity(y)
# it is already stationary data

# but it was not then we have to perform some processing, anyways we are applying it here too.



y_log = np.log(y)



s_data = y_log - y_log.shift(1)

s_data = s_data.dropna()

test_stationarity(s_data)
#create training data



train_prophet = pd.DataFrame()

train_prophet['ds'] = y.index

train_prophet['y'] = y.Open.values
# create model to predict the future openings



from fbprophet import Prophet



model = Prophet().fit(train_prophet)



future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

model.plot(forecast)

model.plot_components(forecast)
#compare the predicted with original dataset



model.plot(forecast)

plt.plot(y,label='original',color='green',linewidth=0.8)

plt.legend(loc='best')

plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
y = df[['Date','High']]

y = y.set_index('Date')



#create training data



train_prophet = pd.DataFrame()

train_prophet['ds'] = y.index

train_prophet['y'] = y.High.values



model = Prophet().fit(train_prophet)



future = model.make_future_dataframe(periods=90)



forecast = model.predict(future)

model.plot_components(forecast)



#compare the predicted with original dataset



model.plot(forecast)

plt.plot(y,label='original',color='green',linewidth=0.8)

plt.legend(loc='best')

plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
y = df[['Date','Close']]

y = y.set_index('Date')



#create training data



train_prophet = pd.DataFrame()

train_prophet['ds'] = y.index

train_prophet['y'] = y.Close.values



model = Prophet().fit(train_prophet)



future = model.make_future_dataframe(periods=90)



forecast = model.predict(future)

model.plot_components(forecast)



#compare the predicted with original dataset



model.plot(forecast)

plt.plot(y,label='original',color='green',linewidth=0.8)

plt.legend(loc='best')

plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
y = df[['Date','Low']]

y = y.set_index('Date')



#create training data



train_prophet = pd.DataFrame()

train_prophet['ds'] = y.index

train_prophet['y'] = y.Low.values



model = Prophet().fit(train_prophet)



future = model.make_future_dataframe(periods=90)



forecast = model.predict(future)

model.plot_components(forecast)



#compare the predicted with original dataset



model.plot(forecast)

plt.plot(y,label='original',color='green',linewidth=0.8)

plt.legend(loc='best')

plt.show()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
#interactive graph s

from fbprophet.plot import plot_plotly, plot_components_plotly

plot_components_plotly(model,forecast)
plot_plotly(model, forecast)