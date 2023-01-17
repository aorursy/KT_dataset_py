import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pylab import rcParams

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error



import seaborn as sns

worldometers=pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

worldometersnew=worldometers[['Country/Region','TotalCases','TotalDeaths','TotalRecovered','Serious,Critical']].copy()

worldometersnew.head()

worldometersnew.iloc[0:10].style.background_gradient(cmap='Reds')





sns.set_style("whitegrid")

timeseries=pd.read_csv('../input/corona-virus-report/day_wise.csv')

timeseriesnew=timeseries[['Date','New cases']].copy()

timeseriesnew['Date']=pd.to_datetime(timeseriesnew['Date']) 



timeseriesnew.head()

timeseriesnew['month'] = timeseriesnew['Date'].apply(lambda x: x.month)

timeseriesnew.set_index('Date', inplace= True)

timeseriesnew=timeseriesnew.fillna(method='ffill')



timeseriesnew.sample(10)
timeseriesyear=timeseriesnew.copy()

tsf=timeseriesnew.loc['2020-04-16':'2020-07-25'].copy()

tsf.drop('month', axis=1,inplace=True)

plt.figure(figsize=(25,8))

plt.plot(tsf)

plt.title('Daily New Cases')

plt.xlabel('Daily Cases From January 2020 To July 2020 : Each Column Grid Represents Half of a Month')

plt.ylabel('Number Of Cases : Time Series')

plt.show()
tsf['Date'] = tsf.index

df_train = tsf[tsf['Date'] < "2020-07-08"]

df_valid = tsf[tsf['Date'] >= "2020-07-08"]

tsf.head()
model_fbp=Prophet()

modelres=model_fbp.fit(df_train[["Date", "New cases"]].rename(columns={"Date": "ds", "New cases": "y"}))

forecast = model_fbp.predict(df_valid[["Date", "New cases"]].rename(columns={"Date": "ds"}))

df_valid["Forecast Prediction"] = forecast.yhat.values
plt.figure(figsize=(20,8))





plt.plot(df_valid[["New cases"]])

plt.plot(df_valid["Forecast Prediction"],color='Green')

plt.plot(df_train[["New cases"]])

plt.title("New Cases")

plt.legend(['Actual Increase of Cases', 'New Cases Predicted By the Model','Fitted Model on Training set'], loc='upper left')

from pandas.tseries.offsets import DateOffset

future_dates=[tsf.index[-1]+ DateOffset(days=x)for x in range(0,30)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=tsf.columns)

future_datest_df['Date'] = future_datest_df.index

future_datest_df.head(10)
future_df=pd.concat([tsf,future_datest_df])



df_trains = future_df[future_df['Date'] < "2020-07-20"]

df_valids = future_df[future_df['Date'] >= "2020-07-06"]

future_df.tail()
model_fbp=Prophet()

modelres=model_fbp.fit(df_trains[["Date", "New cases"]].rename(columns={"Date": "ds", "New cases": "y"}))

forecast = model_fbp.predict(df_valids[["Date", "New cases"]].rename(columns={"Date": "ds"}))

df_valids["Forecast Prediction"] = forecast.yhat.values
plt.figure(figsize=(20,8))





plt.plot(df_valids["Forecast Prediction"],color='Green')

plt.plot(df_train[["New cases"]])

plt.title("New Cases for the month of August")

plt.legend(['New Cases Predicted By the Model','Fitted Model on Training set'], loc='upper left')

timeseries=pd.read_csv('../input/corona-virus-report/day_wise.csv')

timeseriesnew=timeseries[['Date','New recovered']].copy()



timeseriesnew['Date']=pd.to_datetime(timeseriesnew['Date']) 



timeseriesnew.head()

timeseriesnew['month'] = timeseriesnew['Date'].apply(lambda x: x.month)

timeseriesnew.set_index('Date', inplace= True)

timeseriesnew=timeseriesnew.fillna(method='ffill')



timeseriesyear=timeseriesnew.copy()

tsf=timeseriesnew.loc['2020-04-16':'2020-07-15'].copy()

tsf.drop('month', axis=1,inplace=True)

plt.figure(figsize=(20,8))

plt.plot(tsf)

plt.title('Daily Recovered Cases')

plt.xlabel('Daily Recovering Cases From January 2020 To July 2020 : Each Column Grid Represents Each Month')

plt.ylabel('Number Of Cases : Time Series')



plt.show()
tsf['Date'] = tsf.index

df_train = tsf[tsf['Date'] < "2020-07-07"]

df_valid = tsf[tsf['Date'] >= "2020-07-07"]

tsf.head()
model_fbp=Prophet()

modelres=model_fbp.fit(df_train[["Date", "New recovered"]].rename(columns={"Date": "ds", "New recovered": "y"}))

forecast = model_fbp.predict(df_valid[["Date", "New recovered"]].rename(columns={"Date": "ds"}))

df_valid["Forecast Prediction"] = forecast.yhat.values
plt.figure(figsize=(20,8))



plt.plot(df_valid[["New recovered"]])

plt.plot(df_valid["Forecast Prediction"],color='Green')

plt.plot(df_train[["New recovered"]])

plt.title("New Recovered Cases")

plt.legend(['Actual Recovered Cases', 'Recovered Cases Predicted By the Model','Fitted Model on Training set'], loc='upper left')

from pandas.tseries.offsets import DateOffset

future_dates=[tsf.index[-1]+ DateOffset(days=x)for x in range(0,30)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=tsf.columns)

future_datest_df['Date'] = future_datest_df.index

future_datest_df.head(10)
future_df=pd.concat([tsf,future_datest_df])



df_trains = future_df[future_df['Date'] < "2020-07-15"]

df_valids = future_df[future_df['Date'] >= "2020-07-06"]

future_df.tail()
model_fbp=Prophet()

modelres=model_fbp.fit(df_trains[["Date", "New recovered"]].rename(columns={"Date": "ds", "New recovered": "y"}))

forecast = model_fbp.predict(df_valids[["Date", "New recovered"]].rename(columns={"Date": "ds"}))

df_valids["Forecast Prediction"] = forecast.yhat.values
plt.figure(figsize=(20,8))





plt.plot(df_valids["Forecast Prediction"],color='Green')

plt.plot(df_train[["New recovered"]])

plt.title("New Cases for the month of August")

plt.legend(['New Cases Predicted By the Model','Fitted Model on Training set'], loc='upper left')

timeseries=pd.read_csv('../input/corona-virus-report/day_wise.csv')

timeseriesnew=timeseries[['Date','New deaths']].copy()

timeseriesnew['Date']=pd.to_datetime(timeseriesnew['Date']) 



timeseriesnew.head()

timeseriesnew['month'] = timeseriesnew['Date'].apply(lambda x: x.month)

timeseriesnew.set_index('Date', inplace= True)

timeseriesnew=timeseriesnew.fillna(method='ffill')



timeseriesyear=timeseriesnew.copy()

tsf=timeseriesnew.loc['2020-04-16':'2020-07-15'].copy()

tsf.drop('month', axis=1,inplace=True)

plt.figure(figsize=(25,8))

plt.plot(tsf)

plt.title('Daily New Deaths')

plt.xlabel('Daily Deaths From January 2020 To July 2020 : Each Column Grid Represents Each Month')

plt.ylabel('Number Of Cases : Time Series')

plt.show()
tsf['Date'] = tsf.index

df_train = tsf[tsf['Date'] < "2020-06-30"]

df_valid = tsf[tsf['Date'] >= "2020-06-30"]

tsf.head()
model_fbp=Prophet()

modelres=model_fbp.fit(df_train[["Date",'New deaths']].rename(columns={"Date": "ds", "New deaths": "y"}))

forecast = model_fbp.predict(df_valid[["Date", "New deaths"]].rename(columns={"Date": "ds"}))

df_valid["Forecast Prediction"] = forecast.yhat.values
plt.figure(figsize=(20,8))



plt.plot(df_valid[["New deaths"]])

plt.plot(df_valid["Forecast Prediction"],color='Green')

plt.plot(df_train[["New deaths"]])

plt.title("New Deaths")

plt.legend(['Actual Increase of Deaths', 'New Deaths Predicted By the Model','Fitted Model on Training set'], loc='upper left')

from pandas.tseries.offsets import DateOffset

future_dates=[tsf.index[-1]+ DateOffset(days=x)for x in range(0,30)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=tsf.columns)

future_datest_df['Date'] = future_datest_df.index

future_datest_df.head(10)
future_df=pd.concat([tsf,future_datest_df])



df_trains = future_df[future_df['Date'] < "2020-06-27"]

df_valids = future_df[future_df['Date'] >= "2020-06-27"]

future_df.tail()
model_fbp=Prophet()

modelres=model_fbp.fit(df_trains[["Date", "New deaths"]].rename(columns={"Date": "ds", "New deaths": "y"}))

forecast = model_fbp.predict(df_valids[["Date", "New deaths"]].rename(columns={"Date": "ds"}))

df_valids["Forecast Prediction"] = forecast.yhat.values
plt.figure(figsize=(20,8))





plt.plot(df_valids["Forecast Prediction"],color='Green')

plt.plot(df_train[["New deaths"]])

plt.title("New Deaths Prediction for the month of August")

plt.legend(['New Deaths Predicted By the Model','Fitted Model on Training set'], loc='upper left')
