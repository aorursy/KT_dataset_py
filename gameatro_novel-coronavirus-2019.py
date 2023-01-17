import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

df_conf = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_conf.head()
df_conf["Country/Region"].unique()
conf_Italy = df_conf[df_conf["Country/Region"] == "Italy"]

conf_Italy = conf_Italy.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

# Select data from 31st Jan

conf_Italy = conf_Italy.loc[:,'1/31/20':]

conf_Italy = pd.Series(data=conf_Italy.iloc[0].values,index=pd.to_datetime(conf_Italy.columns))

conf_Italy.head()
plt.figure(figsize=(20,6))

plt.plot(conf_Italy.index,conf_Italy.values)

plt.title("Number of confirmed cases in Italy timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_Italy.index,conf_Italy.values)

plt.xticks(conf_Italy.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 5000),rotation=90)

plt.title("Number of confirmed cases in Italy timeline bar plot")
conf_Italy_pday = np.ones(len(conf_Italy))

conf_Italy_pday[0] = conf_Italy[0]

for i in range(1,len(conf_Italy)):

    conf_Italy_pday[i] = conf_Italy[i] - conf_Italy[i-1]

conf_Italy_pday = pd.Series(data=conf_Italy_pday,index = conf_Italy.index)

conf_Italy_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_Italy_pday.index,conf_Italy_pday.values)

plt.xticks(conf_Italy_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 100),rotation=90)

plt.title("Number of confirmed cases per day in Italy")
conf_US = df_conf[df_conf["Country/Region"] == "US"]

conf_US = conf_US.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

conf_US = pd.Series(data=conf_US.iloc[0].values,index=pd.to_datetime(conf_US.columns))

conf_US.head()
plt.figure(figsize=(20,6))

plt.plot(conf_US.index,conf_US.values)

plt.title("Number of confirmed cases in US timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_US.index,conf_US.values)

plt.xticks(conf_US.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 10000),rotation=90)

plt.title("Number of confirmed cases in US timeline bar plot")
conf_US_pday = np.ones(len(conf_US))

conf_US_pday[0] = conf_US[0]

for i in range(1,len(conf_US)):

    conf_US_pday[i] = conf_US[i] - conf_US[i-1]

conf_US_pday = pd.Series(data=conf_US_pday,index = conf_US.index)

conf_US_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_US_pday.index,conf_US_pday.values)

plt.xticks(conf_US_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 1000),rotation=90)

plt.title("Number of confirmed cases per day in US")
conf_SK = df_conf[df_conf["Country/Region"] == "Korea, South"]

conf_SK = conf_SK.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

conf_SK = pd.Series(data=conf_SK.iloc[0].values,index=pd.to_datetime(conf_SK.columns))

conf_SK.head()
plt.figure(figsize=(20,6))

plt.plot(conf_SK.index,conf_SK.values)

plt.title("Number of confirmed cases in South Korea timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_SK.index.date,conf_SK.values)

plt.xticks(conf_SK.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 500),rotation=90)

plt.title("Number of confirmed cases in SK timeline bar plot")
conf_SK_pday = np.ones(len(conf_US))

conf_SK_pday[0] = conf_SK[0]

for i in range(1,len(conf_SK)):

    conf_SK_pday[i] = conf_SK[i] - conf_SK[i-1]

conf_SK_pday = pd.Series(data=conf_SK_pday,index = conf_SK.index)

conf_SK_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_SK_pday.index.date,conf_SK_pday.values)

plt.xticks(conf_SK_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 10),rotation=90)

plt.title("Number of confirmed cases per day in South Korea")
conf_Ind = df_conf[df_conf["Country/Region"] == "India"]

conf_Ind = conf_Ind.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

# Select from 31st Jan

conf_Ind = conf_Ind.loc[:,'1/31/20':]

conf_Ind = pd.Series(data=conf_Ind.iloc[0].values,index=pd.to_datetime(conf_Ind.columns))



conf_Ind.head()
plt.figure(figsize=(20,6))

plt.plot(conf_Ind.index,conf_Ind.values)

plt.title("Number of confirmed cases in India timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_Ind.index.date,conf_Ind.values)

plt.xticks(conf_Ind.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 100),rotation=90)

plt.title("Number of confirmed cases in India timeline bar plot")
conf_Ind_pday = np.ones(len(conf_Ind))

conf_Ind_pday[0] = conf_Ind[0]

for i in range(1,len(conf_Ind)):

    conf_Ind_pday[i] = conf_Ind[i] - conf_Ind[i-1]

conf_Ind_pday = pd.Series(data=conf_Ind_pday,index = conf_Ind.index)

conf_Ind_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_Ind_pday.index.date,conf_Ind_pday.values)

plt.xticks(conf_Ind_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 50),rotation=90)

plt.title("Number of confirmed cases per day in India")
conf_world = df_conf.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

conf_world = conf_world.sum()

conf_world.index = pd.to_datetime(conf_world.index)

conf_world.head()
plt.figure(figsize=(20,6))

plt.plot(conf_world.index,conf_world.values)

plt.title("Number of confirmed cases in world timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_world.index.date,conf_world.values)

plt.xticks(conf_world.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 50000),rotation=90)

plt.title("Number of confirmed cases in world timeline bar plot")
conf_world_pday = np.ones(len(conf_world))

conf_world_pday[0] = conf_world[0]

for i in range(1,len(conf_world)):

    conf_world_pday[i] = conf_world[i] - conf_world[i-1]

conf_world_pday = pd.Series(data=conf_world_pday,index = conf_world.index)

conf_world_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_world_pday.index.date,conf_world_pday.values)

plt.xticks(conf_world.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 3000),rotation=90)

plt.title("Number of confirmed cases per day in world")
import statsmodels.api as sm

from keras.models import Sequential

from keras.layers import LSTM,Dense, TimeDistributed, ConvLSTM2D, Flatten, MaxPooling1D

from keras.layers import Dropout, Activation

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error as mse

x = np.arange(len(conf_world)).reshape(-1,1)

y = conf_world.values
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
# Compare prediction for current values with actual

from datetime import timedelta

test = np.arange(len(x)).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [conf_world.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(conf_world.index,conf_world.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
err = mse(conf_world.values, prediction.values)

print("Training mean squared error = {}".format(err))
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [conf_world.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.title("Future count forecast")

plt.legend(["future count"])

plt.show()
_,ax = plt.subplots(figsize=(20,6))

plt.bar(prediction.index,prediction.values)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 100000),rotation=90)

plt.title("Future count")
x = np.arange(len(conf_world_pday)).reshape(-1,1)

y = conf_world_pday.values
model = MLPRegressor(hidden_layer_sizes=[35, 40,10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
test = np.arange(len(x)).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [conf_world_pday.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(conf_world_pday.index,conf_world_pday.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [conf_world_pday.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)
_,ax = plt.subplots(figsize=(20,6))

plt.bar(prediction.index,prediction.values)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 1000),rotation=90)

plt.title("Future count")
pr_data= pd.DataFrame(conf_world)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)

# compare actual vs predicted

pred_date = pd.DataFrame(conf_world.index)

pred_date.columns = ['ds']

pred = m.predict(pred_date)

pred['yhat'] = np.exp(pred['yhat']) - 1

plt.figure(figsize=(20,6))

plt.plot(conf_world.index, pred.yhat)

plt.plot(conf_world.index, conf_world.values)

plt.legend(['predicted count','actual count'])

plt.show()
err = mse(conf_world.values, pred.yhat)

print("Training mean squared error = {}".format(err))
future=pd.DataFrame([conf_world.index[-1] + timedelta(i+1) for i in range(100)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future counts")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
pr_data= pd.DataFrame(conf_world_pday)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)
# compare actual vs predicted

pred_date = pd.DataFrame(conf_world_pday.index)

pred_date.columns = ['ds']

pred = m.predict(pred_date)

pred['yhat'] = np.exp(pred['yhat']) - 1

plt.figure(figsize=(20,6))

plt.plot(conf_world_pday.index, pred.yhat)

plt.plot(conf_world_pday.index, conf_world_pday.values)

plt.legend(['predicted count','actual count'])

plt.show()
future=pd.DataFrame([conf_world_pday.index[-1] + timedelta(i+1) for i in range(100)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future counts")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count per day')

confirm_cs = pd.DataFrame(conf_world)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
!pip install pmdarima
from pmdarima import auto_arima



stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
#model = ARIMA(arima_data['count'].values, order=(1, 2, 1))

model= SARIMAX(arima_data['count'],order=(1,2,0),seasonal_order=(0,1,1,12)) #Change the model as per the result of above as the dataset is updated

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.title('Forecast vs Actual')

err = mse(conf_world.values, pred)

print("Training mean squared error = {}".format(err))
forcast = fit_model.forecast(steps=90)

pred_y = forcast

plt.figure(figsize=(20,6))

plt.plot(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.show()
confirm_cs = pd.DataFrame(conf_world_pday)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
#model = ARIMA(arima_data['count'].values, order=(2, 1, 2))

model= SARIMAX(arima_data['count'],order=(1,1,0),seasonal_order=(0,1,1,12)) # Change the order as per above result

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.legend(['forecast','actual'])

plt.title('Forecast vs Actual')
forcast = fit_model.forecast(steps=90)

pred_y = forcast

_,ax = plt.subplots(figsize=(20,6))

plt.bar(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height().round().astype(int)), (p.get_x() , p.get_height() + 10000),rotation=90)

plt.show()
df_death = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
df_death.head()
df_death["Country/Region"].unique()
conf_Italy = df_death[df_death["Country/Region"] == "Italy"]

conf_Italy = conf_Italy.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

# Select data from 31st Jan

conf_Italy = conf_Italy.loc[:,'1/31/20':]

conf_Italy = pd.Series(data=conf_Italy.iloc[0].values,index=pd.to_datetime(conf_Italy.columns))

conf_Italy.head()
plt.figure(figsize=(20,6))

plt.plot(conf_Italy.index,conf_Italy.values)

plt.title("Number of deaths in Italy timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_Italy.index,conf_Italy.values)

plt.xticks(conf_Italy.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 500),rotation=90)

plt.title("Number of deaths in Italy timeline bar plot")
conf_Italy_pday = np.ones(len(conf_Italy))

conf_Italy_pday[0] = conf_Italy[0]

for i in range(1,len(conf_Italy)):

    conf_Italy_pday[i] = conf_Italy[i] - conf_Italy[i-1]

conf_Italy_pday = pd.Series(data=conf_Italy_pday,index = conf_Italy.index)

conf_Italy_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_Italy_pday.index,conf_Italy_pday.values)

plt.xticks(conf_Italy_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 10),rotation=90)

plt.title("Number of deaths per day in Italy")
conf_US = df_death[df_death["Country/Region"] == "US"]

conf_US = conf_US.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

conf_US = pd.Series(data=conf_US.iloc[0].values,index=pd.to_datetime(conf_US.columns))

conf_US.head()
plt.figure(figsize=(20,6))

plt.plot(conf_US.index,conf_US.values)

plt.title("Number of deaths in US timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_US.index,conf_US.values)

plt.xticks(conf_US.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 1000),rotation=90)

plt.title("Number of deaths in US timeline bar plot")
conf_US_pday = np.ones(len(conf_US))

conf_US_pday[0] = conf_US[0]

for i in range(1,len(conf_US)):

    conf_US_pday[i] = conf_US[i] - conf_US[i-1]

conf_US_pday = pd.Series(data=conf_US_pday,index = conf_US.index)

conf_US_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_US_pday.index,conf_US_pday.values)

plt.xticks(conf_US_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 100),rotation=90)

plt.title("Number of deaths per day in US")
conf_SK = df_death[df_death["Country/Region"] == "Korea, South"]

conf_SK = conf_SK.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

conf_SK = pd.Series(data=conf_SK.iloc[0].values,index=pd.to_datetime(conf_SK.columns))

conf_SK.head()
plt.figure(figsize=(20,6))

plt.plot(conf_SK.index,conf_SK.values)

plt.title("Number of deaths in South Korea timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_SK.index.date,conf_SK.values)

plt.xticks(conf_SK.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 5),rotation=90)

plt.title("Number of deaths in SK timeline bar plot")
conf_SK_pday = np.ones(len(conf_US))

conf_SK_pday[0] = conf_SK[0]

for i in range(1,len(conf_SK)):

    conf_SK_pday[i] = conf_SK[i] - conf_SK[i-1]

conf_SK_pday = pd.Series(data=conf_SK_pday,index = conf_SK.index)

conf_SK_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_SK_pday.index.date,conf_SK_pday.values)

plt.xticks(conf_SK_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height()+0.5),rotation=90)

plt.title("Number of deaths per day in South Korea")
conf_Ind = df_death[df_death["Country/Region"] == "India"]

conf_Ind = conf_Ind.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

# Select from 31st Jan

conf_Ind = conf_Ind.loc[:,'1/31/20':]

conf_Ind = pd.Series(data=conf_Ind.iloc[0].values,index=pd.to_datetime(conf_Ind.columns))



conf_Ind.head()
plt.figure(figsize=(20,6))

plt.plot(conf_Ind.index,conf_Ind.values)

plt.title("Number of deaths in India timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(conf_Ind.index.date,conf_Ind.values)

plt.xticks(conf_Ind.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 10),rotation=90)

plt.title("Number of deaths in India timeline bar plot")
conf_Ind_pday = np.ones(len(conf_Ind))

conf_Ind_pday[0] = conf_Ind[0]

for i in range(1,len(conf_Ind)):

    conf_Ind_pday[i] = conf_Ind[i] - conf_Ind[i-1]

conf_Ind_pday = pd.Series(data=conf_Ind_pday,index = conf_Ind.index)

conf_Ind_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(conf_Ind_pday.index.date,conf_Ind_pday.values)

plt.xticks(conf_Ind_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 0.5),rotation=90)

plt.title("Number of deaths per day in India")
death_world = df_death.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

death_world = death_world.sum()

death_world.index = pd.to_datetime(death_world.index)

death_world.head()
plt.figure(figsize=(20,6))

plt.plot(death_world.index,death_world.values)

plt.title("Number of deaths in world timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(death_world.index.date,death_world.values)

plt.xticks(death_world.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 5000),rotation=90)

plt.title("Number of deaths in world timeline bar plot")
death_world_pday = np.ones(len(conf_world))

death_world_pday[0] = conf_world[0]

for i in range(1,len(conf_world)):

    conf_world_pday[i] = conf_world[i] - conf_world[i-1]

death_world_pday = pd.Series(data=conf_world_pday,index = conf_world.index)

death_world_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(death_world_pday.index.date,death_world_pday.values)

plt.xticks(death_world.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 300),rotation=90)

plt.title("Number of deaths per day in world")
import statsmodels.api as sm

from keras.models import Sequential

from keras.layers import LSTM,Dense, TimeDistributed, Conv1D, Flatten, MaxPooling1D

from keras.layers import Dropout, Activation

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from statsmodels.tsa.arima_model import ARIMA

x = np.arange(len(death_world)).reshape(-1,1)

y = death_world.values
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
# Compare prediction for current values with actual

from datetime import timedelta

test = np.arange(len(x)).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [death_world.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(death_world.index,death_world.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
err = mse(death_world.values, prediction.values)

print("Training mean squared error = {}".format(err))
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [death_world.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.title("Future count forecast")

plt.legend(["future count"])

plt.show()
_,ax = plt.subplots(figsize=(20,6))

plt.bar(prediction.index,prediction.values)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 10000),rotation=90)

plt.title("Future count")
x = np.arange(len(death_world_pday)).reshape(-1,1)

y = conf_world_pday.values
model = MLPRegressor(hidden_layer_sizes=[35, 40,10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
test = np.arange(len(x)+7).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [death_world_pday.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(death_world_pday.index,conf_world_pday.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [death_world_pday.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)
_,ax = plt.subplots(figsize=(20,6))

plt.bar(prediction.index,prediction.values)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 1000),rotation=90)

plt.title("Future count")
pr_data= pd.DataFrame(death_world)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)
#compare actual vs predicted values

pred_dates = pd.DataFrame(death_world.index)

pred_dates.columns = ['ds']

pred = m.predict(pred_dates)

pred.yhat = np.exp(pred.yhat) - 1

plt.figure(figsize=(20,6))

plt.plot(pred.ds, pred.yhat)

plt.plot(death_world.index,death_world.values)

plt.title("Predicted vs Actual values")

plt.legend(['Predcted','Actual'])

plt.show()
err = mse(death_world.values, pred.yhat)

print("Training mean squared error = {}".format(err))
future=pd.DataFrame([death_world.index[-1] + timedelta(i+1) for i in range(90)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future count")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
pr_data= pd.DataFrame(death_world_pday)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)
#compare actual vs predicted values

pred_dates = pd.DataFrame(death_world_pday.index)

pred_dates.columns = ['ds']

pred = m.predict(pred_dates)

pred.yhat = np.exp(pred.yhat) - 1

plt.figure(figsize=(20,6))

plt.plot(pred.ds, pred.yhat)

plt.plot(death_world_pday.index,death_world_pday.values)

plt.title("Predicted vs Actual values")

plt.legend(['Predcted','Actual'])

plt.show()
err = mse(death_world.values, pred.yhat)

print("Training mean squared error = {}".format(err))


future=pd.DataFrame([death_world_pday.index[-1] + timedelta(i+1) for i in range(90)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future count")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count per day')

confirm_cs = pd.DataFrame(death_world)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
#model = ARIMA(arima_data['count'].values, order=(1, 2, 1))

model= SARIMAX(arima_data['count'],order=(0,2,0),seasonal_order=(0,1,1,12))

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.title('Forecast vs Actual')

err = mse(death_world.values, pred)

print("Training mean squared error = {}".format(err))
forcast = fit_model.forecast(steps=90)

pred_y = forcast

plt.figure(figsize=(20,6))

plt.plot(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.show()
confirm_cs = pd.DataFrame(death_world_pday)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
model= SARIMAX(arima_data['count'],order=(1,1,0),seasonal_order=(0,1,1,12))

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.legend(['forecast','actual'])

plt.title('Forecast vs Actual')
forcast = fit_model.forecast(steps=90)

pred_y = forcast

_,ax = plt.subplots(figsize=(20,6))

plt.bar(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height().round().astype(int)), (p.get_x() , p.get_height() + 10000),rotation=90)

plt.show()



train_data = death_world.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)

n_input = 5

train_x, train_y = split_sequence(scaled_train_data,n_input)

n_features =1

train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],n_features))

for i in range(len(train_x)):

    print(train_x[i],train_y[i])
lstm_model = Sequential()

lstm_model.add(LSTM(input_shape=(n_input, n_features),units=100,activation='relu',return_sequences=True))

#lstm_model.add(Dropout(0.05))

lstm_model.add(LSTM(128))

#lstm_model.add(Dropout(0.05))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.fit(train_x,train_y, epochs = 200)
predicted_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)-5):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    print(current_batch,lstm_pred)

    predicted_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
prediction = pd.Series(data=scaler.inverse_transform(predicted_data).reshape(1,-1)[0].round().astype(int),index=conf_world[5:].index)

prediction
death_world
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.plot(death_world[5:].index,conf_world[5:].values)

plt.title("Prediction vs Actual count")

plt.legend(["prediction","actual"])

plt.show()
future_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)+115):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    future_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
pred_ind = [conf_world_pday.index[0] + timedelta(days=i) for i in range(len(future_data))]

prediction = pd.Series(data=scaler.inverse_transform(future_data).reshape(1,-1)[0].round().astype(int),index=pred_ind)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.show()
train_data = death_world_pday.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
n_input = 5

train_x, train_y = split_sequence(scaled_train_data,n_input)

n_features =1

train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],n_features))

for i in range(len(train_x)):

    print(train_x[i],train_y[i])
lstm_model = Sequential()

lstm_model.add(LSTM(input_shape=(n_input, n_features),units=50,activation='relu'))

#lstm_model.add(Dropout(0.05))

#lstm_model.add(LSTM(256))

#lstm_model.add(Dropout(0.05))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.fit(train_x,train_y, epochs = 200)
predicted_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)-5):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    print(current_batch,lstm_pred)

    predicted_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
prediction = pd.Series(data=scaler.inverse_transform(predicted_data).reshape(1,-1)[0].round().astype(int),index=conf_world[5:].index)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.plot(death_world_pday[5:].index,death_world_pday[5:].values)

plt.title("Prediction vs Actual count")

plt.legend(["prediction","actual"])

plt.show()
df_recov = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
df_recov.head()
df_recov["Country/Region"].unique()
recov_Italy = df_recov[df_recov["Country/Region"] == "Italy"]

recov_Italy = recov_Italy.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

# Select data from 31st Jan

recov_Italy = recov_Italy.loc[:,'1/31/20':]

recov_Italy = pd.Series(data=recov_Italy.iloc[0].values,index=pd.to_datetime(recov_Italy.columns))

recov_Italy.head()
plt.figure(figsize=(20,6))

plt.plot(recov_Italy.index,recov_Italy.values)

plt.title("Number of recovered in Italy timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(recov_Italy.index,recov_Italy.values)

plt.xticks(recov_Italy.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 500),rotation=90)

plt.title("Number of recovered in Italy timeline bar plot")
recov_Italy_pday = np.ones(len(recov_Italy))

recov_Italy_pday[0] = recov_Italy[0]

for i in range(1,len(recov_Italy)):

    recov_Italy_pday[i] = recov_Italy[i] - recov_Italy[i-1]

recov_Italy_pday = pd.Series(data=recov_Italy_pday,index = recov_Italy.index)

recov_Italy_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(recov_Italy_pday.index,recov_Italy_pday.values)

plt.xticks(recov_Italy_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 100),rotation=90)

plt.title("Number of recovered per day in Italy")
recov_US = df_recov[df_recov["Country/Region"] == "US"]

recov_US = recov_US.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

recov_US = pd.Series(data=recov_US.iloc[0].values,index=pd.to_datetime(recov_US.columns))

recov_US.head()
plt.figure(figsize=(20,6))

plt.plot(recov_US.index,conf_US.values)

plt.title("Number of recovered in US timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(recov_US.index,conf_US.values)

plt.xticks(recov_US.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 1000),rotation=90)

plt.title("Number of recovered in US timeline bar plot")
recov_US_pday = np.ones(len(recov_US))

recov_US_pday[0] = recov_US[0]

for i in range(1,len(recov_US)):

    recov_US_pday[i] = recov_US[i] - recov_US[i-1]

recov_US_pday = pd.Series(data=recov_US_pday,index = recov_US.index)

recov_US_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(recov_US_pday.index,recov_US_pday.values)

plt.xticks(recov_US_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 500),rotation=90)

plt.title("Number of recovered per day in US")
recov_SK = df_recov[df_recov["Country/Region"] == "Korea, South"]

recov_SK = recov_SK.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

recov_SK = pd.Series(data=recov_SK.iloc[0].values,index=pd.to_datetime(recov_SK.columns))

recov_SK.head()
plt.figure(figsize=(20,6))

plt.plot(recov_SK.index,recov_SK.values)

plt.title("Number of recovered in South Korea timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(recov_SK.index.date,recov_SK.values)

plt.xticks(recov_SK.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 500),rotation=90)

plt.title("Number of recovered in SK timeline bar plot")
recov_SK_pday = np.ones(len(recov_US))

recov_SK_pday[0] = recov_SK[0]

for i in range(1,len(recov_SK)):

    recov_SK_pday[i] = recov_SK[i] - recov_SK[i-1]

recov_SK_pday = pd.Series(data=recov_SK_pday,index = recov_SK.index)

recov_SK_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(recov_SK_pday.index.date,recov_SK_pday.values)

plt.xticks(recov_SK_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height()+0.5),rotation=90)

plt.title("Number of recovered per day in South Korea")
recov_Ind = df_recov[df_death["Country/Region"] == "India"]

recov_Ind = recov_Ind.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

# Select from 31st Jan

recov_Ind = recov_Ind.loc[:,'1/31/20':]

recov_Ind = pd.Series(data=recov_Ind.iloc[0].values,index=pd.to_datetime(recov_Ind.columns))



conf_Ind.head()
plt.figure(figsize=(20,6))

plt.plot(recov_Ind.index,recov_Ind.values)

plt.title("Number of recovered in India timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(recov_Ind.index.date,recov_Ind.values)

plt.xticks(recov_Ind.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 10),rotation=90)

plt.title("Number of recovered in India timeline bar plot")
recov_Ind_pday = np.ones(len(recov_Ind))

recov_Ind_pday[0] = recov_Ind[0]

for i in range(1,len(recov_Ind)):

    recov_Ind_pday[i] = recov_Ind[i] - recov_Ind[i-1]

recov_Ind_pday = pd.Series(data=recov_Ind_pday,index = recov_Ind.index)

recov_Ind_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(recov_Ind_pday.index.date,recov_Ind_pday.values)

plt.xticks(recov_Ind_pday.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 10),rotation=90)

plt.title("Number of recovered per day in India")
recov_world = df_recov.drop(['Province/State','Country/Region','Lat','Long'],axis=1)

recov_world = recov_world.sum()

recov_world.index = pd.to_datetime(recov_world.index)

recov_world.head()
plt.figure(figsize=(20,6))

plt.plot(recov_world.index,recov_world.values)

plt.title("Number of recovered in world timeline")
_,ax = plt.subplots(figsize=(20,6))

plt.bar(recov_world.index.date,death_world.values)

plt.xticks(recov_world.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 5000),rotation=90)

plt.title("Number of recovered in world timeline bar plot")
recov_world_pday = np.ones(len(recov_world))

recov_world_pday[0] = recov_world[0]

for i in range(1,len(recov_world)):

    recov_world_pday[i] = recov_world[i] - recov_world[i-1]

recov_world_pday = pd.Series(data=recov_world_pday,index = recov_world.index)

recov_world_pday.head()
_,ax = plt.subplots(figsize=(20,6)) 

plt.bar(recov_world_pday.index.date,recov_world_pday.values)

plt.xticks(recov_world.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height())[:-2], (p.get_x() , p.get_height() + 300),rotation=90)

plt.title("Number of recovered per day in world")
import statsmodels.api as sm

from keras.models import Sequential

from keras.layers import LSTM,Dense, TimeDistributed, Conv1D, Flatten, MaxPooling1D

from keras.layers import Dropout, Activation

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from statsmodels.tsa.arima_model import ARIMA

x = np.arange(len(recov_world)).reshape(-1,1)

y = recov_world.values
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
# Compare prediction for current values with actual

from datetime import timedelta

test = np.arange(len(x)).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [recov_world.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(recov_world.index,recov_world.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
err = mse(recov_world.values, prediction.values)

print("Training mean squared error = {}".format(err))
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [recov_world.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.title("Future count forecast")

plt.legend(["future count"])

plt.show()
_,ax = plt.subplots(figsize=(20,6))

plt.bar(prediction.index,prediction.values)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 10000),rotation=90)

plt.title("Future count")
x = np.arange(len(recov_world_pday)).reshape(-1,1)

y = recov_world_pday.values
model = MLPRegressor(hidden_layer_sizes=[35, 40,10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
test = np.arange(len(x)+7).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [recov_world_pday.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(recov_world_pday.index,recov_world_pday.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [recov_world_pday.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)
_,ax = plt.subplots(figsize=(20,6))

plt.bar(prediction.index,prediction.values)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() , p.get_height() + 1000),rotation=90)

plt.title("Future count")
pr_data= pd.DataFrame(recov_world)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)
#compare actual vs predicted values

pred_dates = pd.DataFrame(recov_world.index)

pred_dates.columns = ['ds']

pred = m.predict(pred_dates)

pred.yhat = np.exp(pred.yhat) - 1

plt.figure(figsize=(20,6))

plt.plot(pred.ds, pred.yhat)

plt.plot(recov_world.index,recov_world.values)

plt.title("Predicted vs Actual values")

plt.legend(['Predcted','Actual'])

plt.show()
err = mse(recov_world.values, pred.yhat)

print("Training mean squared error = {}".format(err))
future=pd.DataFrame([recov_world.index[-1] + timedelta(i+1) for i in range(90)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future count")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
pr_data= pd.DataFrame(recov_world_pday)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)
#compare actual vs predicted values

pred_dates = pd.DataFrame(recov_world_pday.index)

pred_dates.columns = ['ds']

pred = m.predict(pred_dates)

pred.yhat = np.exp(pred.yhat) - 1

plt.figure(figsize=(20,6))

plt.plot(pred.ds, pred.yhat)

plt.plot(recov_world_pday.index,recov_world_pday.values)

plt.title("Predicted vs Actual values")

plt.legend(['Predcted','Actual'])

plt.show()
future=pd.DataFrame([recov_world_pday.index[-1] + timedelta(i+1) for i in range(90)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future count")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count per day')

confirm_cs = pd.DataFrame(recov_world)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
model= SARIMAX(arima_data['count'],order=(0,2,1),seasonal_order=(0,1,1,12))

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.legend(['forecast','actual'])

plt.title('Forecast vs Actual')

err = mse(recov_world.values, pred)

print("Training mean squared error = {}".format(err))
forcast = fit_model.forecast(steps=90)

pred_y = forcast

plt.figure(figsize=(20,6))

plt.plot(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.show()
confirm_cs = pd.DataFrame(recov_world_pday)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
model= SARIMAX(arima_data['count'],order=(0,1,1),seasonal_order=(0,1,1,12))

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.legend(['forecast','actual'])

plt.title('Forecast vs Actual')
forcast = fit_model.forecast(steps=90)

pred_y = forcast

_,ax = plt.subplots(figsize=(20,6))

plt.bar(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.xticks(prediction.index, rotation=90)

for p in ax.patches:

    ax.annotate(str(p.get_height().round().astype(int)), (p.get_x() , p.get_height() + 10000),rotation=90)

plt.show()


train_data = recov_world.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)

n_input = 5

train_x, train_y = split_sequence(scaled_train_data,n_input)

n_features =1

train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],n_features))

for i in range(len(train_x)):

    print(train_x[i],train_y[i])
lstm_model = Sequential()

lstm_model.add(LSTM(input_shape=(n_input, n_features),units=100,activation='relu',return_sequences=True))

#lstm_model.add(Dropout(0.05))

lstm_model.add(LSTM(128))

#lstm_model.add(Dropout(0.05))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.fit(train_x,train_y, epochs = 200)
predicted_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)-5):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    print(current_batch,lstm_pred)

    predicted_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
prediction = pd.Series(data=scaler.inverse_transform(predicted_data).reshape(1,-1)[0].round().astype(int),index=recov_world[5:].index)

prediction
recov_world
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.plot(recov_world[5:].index,recov_world[5:].values)

plt.title("Prediction vs Actual count")

plt.legend(["prediction","actual"])

plt.show()
future_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)+115):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    future_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
pred_ind = [recov_world_pday.index[0] + timedelta(days=i) for i in range(len(future_data))]

prediction = pd.Series(data=scaler.inverse_transform(future_data).reshape(1,-1)[0].round().astype(int),index=pred_ind)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.show()
train_data = recov_world_pday.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
n_input = 5

train_x, train_y = split_sequence(scaled_train_data,n_input)

n_features =1

train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],n_features))

for i in range(len(train_x)):

    print(train_x[i],train_y[i])
lstm_model = Sequential()

lstm_model.add(LSTM(input_shape=(n_input, n_features),units=50,activation='relu',return_sequences=True))

#lstm_model.add(Dropout(0.05))

lstm_model.add(LSTM(256))

#lstm_model.add(Dropout(0.05))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.fit(train_x,train_y, epochs = 200)
predicted_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)-5):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    print(current_batch,lstm_pred)

    predicted_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
prediction = pd.Series(data=scaler.inverse_transform(predicted_data).reshape(1,-1)[0].round().astype(int),index=recov_world[5:].index)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.plot(recov_world_pday[5:].index,recov_world_pday[5:].values)

plt.title("Prediction vs Actual count")

plt.legend(["prediction","actual"])

plt.show()
act_world = conf_world - (death_world + recov_world)

act_world
plt.figure(figsize=(20,6))

plt.bar(act_world.index,conf_world.values, label='Active',color='r')

plt.bar(recov_world.index, recov_world.values, label='Recovered', color='b')

plt.bar(death_world.index, death_world.values, label='Dead', color='black')

plt.xlabel('Date')

plt.ylabel('Count')

plt.legend(frameon=True, fontsize=12)

plt.xticks(act_world.index, rotation=90)

plt.title('Active vs Recovered vs Dead',fontsize=30)

plt.show()
x = np.arange(len(act_world)).reshape(-1,1)

y = act_world.values
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[32, 40, 10], max_iter=50000, alpha=0.0005, random_state=26)

model.fit(x, y)
from datetime import timedelta

test = np.arange(len(x)).reshape(-1, 1)

pred = model.predict(test)

pred = pred.round().astype(int)

pred_time = [act_world.index[0] + timedelta(days=i) for i in range(len(pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(pred,pred_time)
plt.figure(figsize=(20,6))

plt.plot(act_world.index,act_world.values)

plt.plot(prediction.index,prediction.values)

plt.title("Actual count vs Predicted count")

plt.legend(["actual count","predicted count"])

plt.show()
err = mse(recov_world.values, prediction.values)

print("Training mean squared error = {}".format(err))
# Forecast future count

future = np.arange(len(x),len(x)+90).reshape(-1, 1)

future_pred = model.predict(future)

future_pred = future_pred.round().astype(int)

pred_time = [act_world.index[-1] + timedelta(days=i) for i in range(len(future_pred))]

pred_time = pd.to_datetime(pred_time)

prediction = pd.Series(future_pred,pred_time)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.title("Future count forecast")

plt.legend(["future count"])

plt.show()
pr_data= pd.DataFrame(act_world)

pr_data = pr_data.reset_index()

pr_data.columns = ['ds','y']

pr_data.head()
pr_data['y'] = np.log(pr_data['y'] + 1)

m=Prophet()

m.fit(pr_data)
# compare actual vs predicted

pred_date = pd.DataFrame(act_world.index)

pred_date.columns = ['ds']

pred = m.predict(pred_date)

pred['yhat'] = np.exp(pred['yhat']) - 1

plt.figure(figsize=(20,6))

plt.plot(act_world.index, pred.yhat)

plt.plot(act_world.index, act_world.values)

plt.legend(['predicted count','actual count'])

plt.show()
err = mse(act_world.values, pred.yhat)

print("Training mean squared error = {}".format(err))
future=pd.DataFrame([act_world.index[-1] + timedelta(i+1) for i in range(100)])

future.columns = ['ds']

forecast=m.predict(future)

forecast['yhat'] = np.exp(forecast['yhat']) - 1

forecast
plt.figure(figsize=(20,6))

plt.plot(forecast.ds, forecast.yhat)

plt.title("Future counts")

plt.show()
figure = plot_plotly(m, forecast)

py.iplot(figure,image_width=800) 

figure = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
confirm_cs = pd.DataFrame(act_world)

arima_data = confirm_cs.reset_index()

arima_data.columns = ['confirmed_date','count']

arima_data.head()
stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 

                          max_p = 3, max_q = 3, m = 12, 

                          start_P = 0, seasonal = True, 

                          d = None, D = 1, trace = True, 

                          error_action ='ignore',    

                          suppress_warnings = True,  

                          stepwise = True)           

stepwise_fit.summary()
model= SARIMAX(arima_data['count'],order=(1,2,0),seasonal_order=(0,1,1,12))

fit_model = model.fit(full_output=True, disp=True)

fit_model.summary()
plt.figure(figsize=(20,6))

pred = fit_model.predict(0,len(arima_data)-1)

plt.plot(arima_data.confirmed_date,pred)

plt.plot(arima_data.confirmed_date,arima_data['count'])

plt.title('Forecast vs Actual')
err = mse(act_world.values, pred)

print("Training mean squared error = {}".format(err))
forcast = fit_model.forecast(steps=90)

pred_y = forcast

plt.figure(figsize=(20,6))

plt.plot(pd.to_datetime([conf_world.index[-1] + timedelta(days=i) for i in range(len(pred_y))]),pred_y)

plt.show()
train_data = act_world.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)

n_input = 5

train_x, train_y = split_sequence(scaled_train_data,n_input)

n_features =1

train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],n_features))
for i in range(len(train_x)):

    print(train_x[i],train_y[i])
lstm_model = Sequential()

lstm_model.add(LSTM(input_shape=(n_input, n_features),units=100,activation='relu',return_sequences=True))

#lstm_model.add(Dropout(0.05))

lstm_model.add(LSTM(256))

#lstm_model.add(Dropout(0.05))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lstm_model.fit(train_x,train_y, epochs = 200)
predicted_data = []

batch = scaled_train_data[:n_input].copy()

current_batch = batch.reshape((1, n_input, n_features))

lstm_pred = lstm_model.predict(current_batch)[0]

for i in range(len(train_data)-5):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    print(current_batch,lstm_pred)

    predicted_data.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
prediction = pd.Series(data=scaler.inverse_transform(predicted_data).reshape(1,-1)[0].round().astype(int),index=conf_world[5:].index)

prediction
plt.figure(figsize=(20,6))

plt.plot(prediction.index,prediction.values)

plt.plot(conf_world[5:].index,conf_world[5:].values)

plt.title("Prediction vs Actual count")

plt.legend(["prediction","actual"])

plt.show()