# Imports



import numpy as np

from numpy import array

import pandas as pd

import datetime

import os



import tensorflow as tf



import plotly.graph_objects as go



import plotly.express as px

import plotly.io as pio

pio.templates.default = "presentation"

from plotly.subplots import make_subplots

import seaborn as sns



from datetime import datetime

from datetime import timedelta



from statsmodels.tsa.arima_model import ARIMA



from fbprophet import Prophet



from statsmodels.tsa.api import Holt



from pandas import DataFrame
# CSV files Paths



base_dir = '/kaggle/input/novel-corona-virus-2019-dataset/'

csv_files_dir = os.path.join(base_dir, 'covid_19_data.csv')
# Imports dataset Convid19



data = pd.read_csv(csv_files_dir)
print('Shape of database : ', data.shape)

data.head(3)
# Set the name of the country for the predictions : 



country_name = "France"



# Set number of days to predict:



numbers_days_predict = 14
# Create new datasets name



covid_19_own = []



# Create datas lists



Confirmed = []

Deaths = []



Evol_confirmed = []

Evol_deaths = []



# Create date list



Date = []
#Functions for data processing



# Function for create new datasets



def create_new_datasets(country):

    

    temp = []

    

    # Loc datas for new dataset

    

    for i in range(len(data)):

        # For country

        if data['Country/Region'][i] == country :

        

        # For region

        #if data['Province/State'][i] == country:

                        

            temp.append(data.loc[i])

                

    temp = pd.DataFrame(temp).reset_index()

    

    return temp



# Function for get values from dataset



def get_all_series_values(dataset, confirmed_list_name, deaths_list_name, date_name):

    

    confirmed, deaths, recovered = 0, 0, 0

    

    for i in range(len(dataset)):

        

        date = dataset['ObservationDate'][i]

        

        confirmed += dataset['Confirmed'][i]

        deaths += dataset['Deaths'][i]

        

        if (i < len(dataset) - 1 and dataset['ObservationDate'][i + 1] != date) or i == len(dataset) - 1:

            

            confirmed_list_name.append(confirmed)

            deaths_list_name.append(deaths)

            date_name.append(date)

            confirmed, deaths,  = 0, 0

            

# Function for get evolution per days values            



def get_evol_values(list_name,evol_name):

    

    i = 0

    

    while i < len(list_name) - 1:

        

        if i == 0:

            

            evol_name.append(0)

            

        else:

            

            temp0 = ((list_name[i+1] - list_name[i]) / list_name[i]) * 100

            evol_name.append(abs(temp0))

            

        i = i + 1      
# Get country/region dataset



covid_19_own = create_new_datasets(country_name)

print("Shape of dataframe : ", covid_19_own.shape)

covid_19_own.head(3)
# Get series list and date list



get_all_series_values(covid_19_own, Confirmed, Deaths, Date)

Date = pd.to_datetime(pd.Series(Date), format='%m/%d/%Y')
fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Confirmed, name = 'Confirmed', marker_color='rgba(64, 246, 44, 1)'))

fig.add_trace(go.Scatter(x=Date, y=Deaths, name = 'Deaths', marker_color='rgba(227, 14, 14, 1)'))



title_status = "Status of " + country_name



fig.update_layout(title=title_status,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
# Get evolution per days lists



get_evol_values(Confirmed, Evol_confirmed)

get_evol_values(Deaths, Evol_deaths)
fig = go.Figure()



fig.add_trace(go.Bar(x=Date, y=Evol_confirmed, name = 'Confirmed', marker_color='rgba(236, 249, 92, 1)'))

fig.add_trace(go.Bar(x=Date, y=Evol_deaths, name = 'Deaths', marker_color='rgba(227, 14, 14, 1)'))



title_evol = "Evolution of new cases per days in " + country_name



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='% of new cases')



fig.show()
# Def results lists of models



ARIMA_confirmed_pred = []

ARIMA_deaths_pred = []

ARIMA_pred_date = []



HOLT_confirmed_pred = []

HOLT_deaths_pred = []

HOLT_pred_date = []



Prophete_confirmed_pred = []

Prophete_deaths_pred = []

Prophete_date_pred = []



CNN_confirmed_pred = []

CNN_deaths_pred = []

CNN_pred_date = []
df_series = pd.DataFrame(Confirmed)

#df_series = pd.DataFrame(Confirmed[:183]) 

X = df_series.values

model = ARIMA(X, order=(5,1,0))

model_fit = model.fit(disp=0)



forecast_confirmed = model_fit.forecast(steps=numbers_days_predict)[0]



last_date = len(Date) - 1



Date_forecast = []



i = 1



while i < len(forecast_confirmed) + 1:

    

    temp = Date[last_date] +  timedelta(days=i)

    #temp = Date[last_date - 14] +  timedelta(days=i)

    Date_forecast.append(temp)

    

    i = i + 1

    

ARIMA_confirmed_pred = forecast_confirmed   

ARIMA_pred_date = Date_forecast



fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Confirmed, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=Date_forecast, y=forecast_confirmed, name = 'Prediction values', marker_color='rgba(214, 10, 10, 1)'))



title_evol = "ARIMA predictions of Confirmed cases in " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
df_series = pd.DataFrame(Deaths) 

#df_series = pd.DataFrame(Deaths[:183]) 

X = df_series.values

model = ARIMA(X, order=(5,1,0))

model_fit = model.fit(disp=0)



forecast_deaths = model_fit.forecast(steps=numbers_days_predict)[0]



last_date = len(Date) - 1



Date_forecast = []



i = 1



while i < len(forecast_deaths) + 1:

    

    temp = Date[last_date] +  timedelta(days=i)

    #temp = Date[last_date - 14] +  timedelta(days=i)

    Date_forecast.append(temp)

    

    i = i + 1

    

fig = go.Figure()



ARIMA_deaths_pred = forecast_deaths



fig.add_trace(go.Scatter(x=Date, y=Deaths, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=Date_forecast, y=forecast_deaths, name = 'Prediction values', marker_color='rgba(214, 10, 10, 1)'))



title_evol = "ARIMA predictions of Deaths cases in " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
df_confirmed = pd.DataFrame(

    {'ds': Date,

     'y': Confirmed

    })



prophet = Prophet()

prophet.fit(df_confirmed)



future = prophet.make_future_dataframe(periods=numbers_days_predict)

forecast = prophet.predict(future)



Prophete_confirmed_pred = forecast['yhat']



Prophete_date_pred = forecast['ds']



min_pred = forecast['yhat_lower']

max_pred = forecast['yhat_upper']



start_grosse = len(Confirmed)



fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Confirmed, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=Prophete_date_pred, y=Prophete_confirmed_pred, name = 'Prediction values', marker_color='rgba(214, 10, 10, .5)'))



title_evol = "PROPHET predictions of Confirmed cases is " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
df_deaths = pd.DataFrame(

    {'ds': Date,

     'y': Deaths

    })



prophet = Prophet()

prophet.fit(df_deaths)



future = prophet.make_future_dataframe(periods=numbers_days_predict)

forecast = prophet.predict(future)



Prophete_deaths_pred = forecast['yhat']



Prophete_date_pred = forecast['ds']



min_pred = forecast['yhat_lower']

max_pred = forecast['yhat_upper']



start_grosse = len(Deaths)



fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Deaths, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=Prophete_date_pred, y=Prophete_deaths_pred, name = 'Prediction values', marker_color='rgba(214, 10, 10, .5)'))



title_evol = "PROPHET predictions of Deaths cases is " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
df_series = pd.DataFrame(Confirmed)

#df_series = pd.DataFrame(Confirmed[:183]) 

X = df_series.values



model = Holt(X)

model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.1,optimized=False)



forecast_confirmed = model_fit.forecast(14)







last_date = len(Date) - 1



Date_forecast = []



i = 1



while i < len(forecast_confirmed) + 1:

    

    temp = Date[last_date] +  timedelta(days=i)

    #temp = Date[last_date - 14] +  timedelta(days=i)

    Date_forecast.append(temp)

    

    i = i + 1

    

HOLT_confirmed_pred = forecast_confirmed

HOLT_pred_date = Date_forecast



fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Confirmed, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=HOLT_confirmed_pred, name = 'Prediction values', marker_color='rgba(214, 10, 10, 1)'))



title_evol = "Holt's Linear predictions of Confirmed cases in " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
df_series = pd.DataFrame(Deaths)

#df_series = pd.DataFrame(Deaths[:183])

X = df_series.values



model = Holt(X)

model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.1,optimized=False)



forecast_confirmed = model_fit.forecast(14)





last_date = len(Date) - 1



Date_forecast = []



i = 1



while i < len(forecast_confirmed) + 1:

    

    temp = Date[last_date] +  timedelta(days=i)

    #temp = Date[last_date -14] +  timedelta(days=i)

    Date_forecast.append(temp)

    

    i = i + 1

    

HOLT_deaths_pred = forecast_confirmed

HOLT_pred_date = Date_forecast



fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Deaths, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=HOLT_deaths_pred, name = 'Prediction values', marker_color='rgba(214, 10, 10, 1)'))



title_evol = "Holt's Linear predictions of Deaths cases in " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
# Function for split sequence



n_steps = 3

n_features = 1



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

    return array(X), array(y)
model = tf.keras.models.Sequential([

    

    tf.keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)),

    tf.keras.layers.MaxPooling1D(pool_size=2),

    

    tf.keras.layers.Flatten(),

    

    tf.keras.layers.Dense(128, activation = 'relu'),

    

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Dense(1, activation = 'relu'),

    

])



model.compile(optimizer='adam', loss='mse')
# define input sequence

#raw_seq = Confirmed

raw_seq = Confirmed

# choose a number of time steps

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]



X = X.reshape((X.shape[0], X.shape[1], n_features))



# fit model

model.fit(X, y, epochs=1000, verbose=0)



# prediction



x_input = array([Confirmed[len(Confirmed) - 3], Confirmed[len(Confirmed) - 2], Confirmed[len(Confirmed) - 1]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)

pred1 = yhat.item()

CNN_confirmed_pred.append(pred1)



x_input = array([Confirmed[len(Confirmed) - 2], Confirmed[len(Confirmed) - 1], CNN_confirmed_pred[0]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)

pred2 = yhat.item()

CNN_confirmed_pred.append(pred2)



x_input = array([Confirmed[len(Confirmed) - 1], CNN_confirmed_pred[0], CNN_confirmed_pred[1]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)

pred3 = yhat.item()

CNN_confirmed_pred.append(pred3)



i = 0



while i < 11:

    

    in1 = CNN_confirmed_pred[i]

    in2 = CNN_confirmed_pred[i+1]

    in3 = CNN_confirmed_pred[i+2]

    

    x_input = array([in1, in2, in3])

    x_input = x_input.reshape((1, n_steps, n_features))

    

    yhat = model.predict(x_input, verbose=2)



    pred = yhat.item()

    

    CNN_confirmed_pred.append(pred)

    

    i = i + 1

fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Confirmed, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=CNN_confirmed_pred, name = 'Prediction values', marker_color='rgba(214, 10, 10, 1)'))



title_evol = "CNN predictions of Confirmed cases in " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
# define input sequence

raw_seq = Deaths

# choose a number of time steps

n_steps = 3

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))



# fit model

model.fit(X, y, epochs=1000, verbose=0)



# prediction



x_input = array([Deaths[len(Deaths) - 3], Deaths[len(Deaths) - 2], Deaths[len(Deaths) - 1]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)

pred1 = yhat.item()

CNN_deaths_pred.append(pred1)



x_input = array([Deaths[len(Deaths) - 2], Deaths[len(Deaths) - 1], CNN_deaths_pred[0]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)

pred2 = yhat.item()

CNN_deaths_pred.append(pred2)



x_input = array([Deaths[len(Deaths) - 1], CNN_deaths_pred[0], CNN_deaths_pred[1]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)

pred3 = yhat.item()

CNN_deaths_pred.append(pred3)



i = 0



while i < 11:

    

    in1 = CNN_deaths_pred[i]

    in2 = CNN_deaths_pred[i+1]

    in3 = CNN_deaths_pred[i+2]

    

    x_input = array([in1, in2, in3])

    x_input = x_input.reshape((1, n_steps, n_features))

    

    yhat = model.predict(x_input, verbose=2)

    

    

    pred = yhat.item()

    

    CNN_deaths_pred.append(pred)

    

    i = i + 1

    
fig = go.Figure()



fig.add_trace(go.Scatter(x=Date, y=Deaths, name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=CNN_deaths_pred, name = 'Prediction values', marker_color='rgba(214, 10, 10, 1)'))



title_evol = "CNN predictions of Deaths cases in " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
count_plot_confirmed = len(Confirmed) - int(0.1 * len(Confirmed))
fig = go.Figure()



fig.add_trace(go.Scatter(x=Date[count_plot_confirmed:len(Confirmed)], y=Confirmed[count_plot_confirmed:len(Confirmed)], name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))



fig.add_trace(go.Scatter(x=ARIMA_pred_date, y=ARIMA_confirmed_pred, mode='lines', name = 'ARIMA prediction values', marker_color='rgba(227, 7, 227, 1)'))

fig.add_trace(go.Scatter(x=Prophete_date_pred[len(Confirmed):len(Prophete_date_pred)], mode='lines', y=Prophete_confirmed_pred[len(Confirmed):len(Prophete_confirmed_pred)], name = 'PROPHETE prediction values', marker_color='rgba(232, 82, 12, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=HOLT_confirmed_pred, mode='lines', name = 'HOLTs prediction values', marker_color='rgba(73, 238, 238, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=CNN_confirmed_pred, mode='lines', name = 'CNN prediction values', marker_color='rgba(249, 213, 51, 1)'))



title_evol = "Comparison of different models for the number of confirmed cases for " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Nombre de cas')



fig.show()
moy_model = []



i = 0



while i < len(ARIMA_confirmed_pred):

    

    a = ARIMA_confirmed_pred[i]

    b = HOLT_confirmed_pred[i]

    c = CNN_confirmed_pred[i]

    d = Prophete_confirmed_pred[i + 197]

    

    moy = (a + b + c + d)/4

    

    moy_model.append(moy)

    

    i = i + 1
fig = go.Figure()



fig.add_trace(go.Scatter(x=Date[count_plot_confirmed:len(Confirmed)], y=Confirmed[count_plot_confirmed:len(Confirmed)], name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))



fig.add_trace(go.Scatter(x=ARIMA_pred_date, y=moy_model, mode='lines', name = 'moy', marker_color='rgba(227, 7, 227, 1)'))



title_evol = "Model average (confirmed cases) for " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Nombre de cas')



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=Date[count_plot_confirmed:len(Deaths)], y=Deaths[count_plot_confirmed:len(Deaths)], name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))



fig.add_trace(go.Scatter(x=ARIMA_pred_date, y=ARIMA_deaths_pred, mode='lines', name = 'ARIMA prediction values', marker_color='rgba(227, 7, 227, 1)'))

fig.add_trace(go.Scatter(x=Prophete_date_pred[len(Confirmed):len(Prophete_date_pred)], mode='lines', y=Prophete_deaths_pred[len(Confirmed):len(Prophete_confirmed_pred)], name = 'PROPHETE prediction values', marker_color='rgba(227, 117, 7, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=HOLT_deaths_pred, mode='lines', name = 'HOLTs prediction values', marker_color='rgba(73, 238, 238, 1)'))

fig.add_trace(go.Scatter(x=HOLT_pred_date, y=CNN_deaths_pred, mode='lines', name = 'CNN prediction values', marker_color='rgba(249, 213, 51, 1)'))



title_evol = "Comparison of different models for the number of death cases for " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()
moy_model = []



i = 0



while i < len(ARIMA_confirmed_pred):

    

    a = ARIMA_deaths_pred[i]

    b = HOLT_deaths_pred[i]

    #c = CNN_deaths_pred[i]

    d = Prophete_deaths_pred[i + 197]

    

    moy = (a + b + d)/3 # (a + b + c + d)/4

    

    moy_model.append(moy)

    

    i = i + 1
fig = go.Figure()



fig.add_trace(go.Scatter(x=Date[count_plot_confirmed:len(Deaths)], y=Deaths[count_plot_confirmed:len(Deaths)], name = 'Real values', marker_color='rgba(29, 240, 29, 1)'))



fig.add_trace(go.Scatter(x=ARIMA_pred_date, y=moy_model, mode='lines', name = 'ARIMA prediction values', marker_color='rgba(227, 7, 227, 1)'))



title_evol = "Model average (death cases) for " + country_name + " (" + str(numbers_days_predict) + " days)"



fig.update_layout(title=title_evol,

                   xaxis_title='Dates',

                   yaxis_title='Number of cases')



fig.show()