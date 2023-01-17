import pandas

import numpy

from keras.preprocessing.sequence import TimeseriesGenerator

import plotly.graph_objects as go

import datetime

from keras.models import Sequential

from keras.layers import LSTM, Dense



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
numpy.random.seed(7)
dataframe = pandas.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

dataframe.head()
dataframe = dataframe.drop(['State/UnionTerritory', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths', 'Time', 'Sno'], axis=1)

dataframe.head()
dataframe['Date'] = pandas.to_datetime(dataframe['Date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d')

dataframe
gdf = dataframe.groupby('Date')

data = []

date = []

cases = []

for name, df in gdf:

    print(name)

    date.append(name)

    s = sum(df['Confirmed'].astype(float))

    cases.append(s)

    data.append([s])

data
len(data)
dataset = numpy.array(data)

dataset.shape
dataset.shape
for i in range(len(cases)-1, 1, -1):

    cases[i] = cases[i] - cases[i-1]

    dataset[i][0] = cases[i]
trace = go.Scatter(

    x = date,

    y = cases,

    mode = 'lines',

    name = 'Data'

)

layout = go.Layout(

    title = "Covid 19 India",

    xaxis = {'title' : "Date"},

    yaxis = {'title' : "Confirmed Cases"}

)

fig = go.Figure(data=[trace], layout=layout)

fig.show()
split_percent = 0.80

split = int(split_percent*len(dataset))

split
dataset_train = dataset[:split]

dataset_test = dataset[split:]



date_train = date[:split]

date_test = date[split:]



print(len(dataset_train))

print(len(dataset_test))
look_back = 3

train_generator = TimeseriesGenerator(dataset_train, dataset_train, length=look_back, batch_size=1)     

test_generator = TimeseriesGenerator(dataset_test, dataset_test, length=look_back, batch_size=1)

train_generator
model = Sequential()

model.add(

    LSTM(10,

        activation='relu',

        return_sequences=True,

        input_shape=(look_back,1))

)

model.add(LSTM(7, return_sequences=True, activation='relu'))

model.add(LSTM(3, activation='relu'))

model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mse')
model.summary()
num_epochs = 200

history = model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
prediction = model.predict_generator(test_generator)
dataset_train = dataset_train.reshape((-1))

dataset_test = dataset_test.reshape((-1))



trace1 = go.Scatter(

    x = date_train,

    y = dataset_train,

    mode = 'lines',

    name = 'Data'

)

trace2 = go.Scatter(

    x = date_test,

    y = prediction,

    mode = 'lines',

    name = 'Prediction'

)

trace3 = go.Scatter(

    x = date_test,

    y = dataset_test,

    mode='lines',

    name = 'Ground Truth'

)

layout = go.Layout(

    title = "Covid 19 India",

    xaxis = {'title' : "Date"},

    yaxis = {'title' : "Confirmed Cases"}

)

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

fig.show()
def predict(num_prediction, model):

    prediction_list = dataset[-look_back:]

    

    for _ in range(num_prediction):

        x = prediction_list[-look_back:]

        x = x.reshape((1, look_back, 1))

        out = model.predict(x)[0][0]

        prediction_list = numpy.append(prediction_list, out)

    prediction_list = prediction_list[look_back-1:]

        

    return prediction_list

    

def predict_dates(num_prediction):

    last_date = date[-1]

    prediction_dates = pandas.date_range(last_date, periods=num_prediction+1).tolist()

    return prediction_dates
num_prediction = 10

forecast = predict(num_prediction, model).astype(int)

forecast_dates = predict_dates(num_prediction)
given_trace = go.Scatter(

    x = date,

    y = cases,

    mode = 'lines',

    name = 'Data' 

)

forcast_trace = go.Scatter(

    x = forecast_dates,

    y = forecast,

    mode = 'lines',

    name = 'Data'

)

layout = go.Layout(

    title = "Covid 19 India Forcast Information",

    xaxis = {'title' : "Date"},

    yaxis = {'title' : "Confirmed Cases"}

)

fig = go.Figure(data=[given_trace, forcast_trace], layout=layout)

fig.show()