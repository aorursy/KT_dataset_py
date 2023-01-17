# imports

import os

from operator import itemgetter

import numpy as np

import pandas as pd

import tensorflow as tf

import plotly.graph_objects as go



# constants

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)

colors = {

  'very_light_gray': '#ececec',

  'light_gray': '#b6b6b6',

  'medium_gray': '#929292',

  'very_dark_gray': '#414141',

  'orange': '#ff6f00',

  'light_blue': '#79c3ff',

  'light_purple': '#d88aff',

  'light_green': '#b4ec70',

  'light_yellow': '#fff27e',

  'light_red': '#ff7482',

  'light_cyan': '#84ffff',

    'red': '#ff0000',

    'blue': '#0000ff',

}

start_date = np.datetime64('2020-01-22')

all_dates = [start_date + np.timedelta64(x, 'D') for x in range(0, 100)]
# converts a country's data into a time series dataframe

def convert_to_ts (data, country):

  df = data[data['Country/Region'] == country].groupby(['Date'], as_index=False)['ConfirmedCases','Fatalities'].sum()

  df['Date'] = df['Date'].astype('datetime64[ns]')

  return df



data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

dat = [

  { 'name': 'China', 'color': 'light_gray' },

  { 'name': 'Korea, South', 'color': 'medium_gray' },

  { 'name': 'Italy', 'color': 'very_dark_gray' },

  { 'name': 'Iran', 'color': 'light_blue' },

  { 'name': 'Spain', 'color': 'light_purple' },

  { 'name': 'Germany', 'color': 'light_green' },

  { 'name': 'France', 'color': 'light_yellow' },

  { 'name': 'United Kingdom', 'color': 'light_red' },

  { 'name': 'Switzerland', 'color': 'light_cyan' },

  { 'name': 'US', 'color': 'orange' },

    { 'name': 'Japan', 'color': 'red' }, 

    { 'name': 'Australia', 'color': 'blue'},

]

countries = { d['name']: convert_to_ts(data, d['name']) for d in dat}
data.tail()
countries['China'].tail()
# Code based on https://www.tensorflow.org/guide/keras/custom_layers_and_models

class Logistic(tf.keras.layers.Layer):



  def __init__(self, units=1):

    super(Logistic, self).__init__()

    self.units = units



  def build(self, input_shape):

    self.w = self.add_weight(shape=(input_shape[-1], self.units),

                             initializer=tf.random_normal_initializer(1.0),

                             trainable=True)

    self.b = self.add_weight(shape=(self.units,),

                             initializer='random_normal',

                             trainable=True)

    self.L = self.add_weight(shape=(self.units,),

                             initializer=tf.random_normal_initializer(1.0),

                             trainable=True,

                             name='L')

    self.k = self.add_weight(shape=(self.units,),

                             initializer=tf.random_normal_initializer(1.0),

                             trainable=True,

                             name='k')

    self.xo = self.add_weight(shape=(self.units,),

                             initializer=tf.random_normal_initializer(0.0),

                             trainable=True,

                             name='xo')

    

  def _logistic (self, x):

    return self.L * self.L / (1 + tf.math.exp(-self.k * (x - self.xo)))



  def call(self, inputs):

    return self._logistic(tf.matmul(inputs, self.w) + self.b)

model = tf.keras.models.Sequential()

model.add(tf.keras.Input(shape=(1,)))

model.add(Logistic(2))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
input_x = np.array([[i] for i in range(0,len(countries['China']))])

input_y = countries['China'][['ConfirmedCases','Fatalities']]/countries['China'][['ConfirmedCases','Fatalities']].max()

history = model.fit(input_x, input_y, epochs=1000)
input_y.head()
for layer in model.layers:

    print(layer.get_weights())
fig = go.Figure()



fig.add_trace(go.Scatter(

    #x = len(,

    y = history.history['loss'],

    name = 'Loss',

  ))



fig.update_layout(title_text='Loss',

                  xaxis_rangeslider_visible=True)



fig.show()

fig = go.Figure()



fig.add_trace(go.Scatter(

    x = countries['China']['Date'],

    y = countries['China']['ConfirmedCases'],

    name = 'Value',

    line_width = 3,

    line_color = '#0000ff'

  ))



fig.add_trace(go.Scatter(

    x = countries['China']['Date'],

    y = countries['China']['Fatalities'],

    name = 'Value',

    line_width = 3,

    line_color = '#ff0000'

  ))



input_x = np.array([[i] for i in range(0,100)])

predictions = model.predict(input_x)



fig.add_trace(go.Scatter(

    x = all_dates,

    y = [p[0] * countries['China']['ConfirmedCases'].max() for p in predictions],

    name = 'ConfirmedCases Prediction',

    line_width = 1,

    line_dash = 'dot',

    line_color = '#0000ff'

  ))



fig.add_trace(go.Scatter(

    x = all_dates,

    y = [p[1] * countries['China']['Fatalities'].max()  for p in predictions],

    name = 'Fatalities Prediction',

    line_width = 1,

    line_dash = 'dot',

    line_color = '#ff0000'

  ))



fig.update_layout(title_text='Prediction estimate',

                  xaxis_rangeslider_visible=True)



fig.show()
for d in dat:

    print('Processing', d['name'])

    country = countries[d['name']]

    #d['model'] = tf.keras.models.Sequential()

    #d['model'].add(tf.keras.Input(shape=(1,)))

    #d['model'].add(Logistic())

    #d['model'].compile(optimizer='adam', loss='mean_squared_error')

    input_x = np.array([[i] for i in range(0,len(country))])

    #input_y = country['ConfirmedCases']/country['ConfirmedCases'].max()

    input_y = country[['ConfirmedCases','Fatalities']]/country[['ConfirmedCases','Fatalities']].max()

    history = model.fit(input_x, input_y, epochs=1000, verbose=0)

    input_x = np.array([[i] for i in range(0,100)])

    #predictions = [i[0] * country['ConfirmedCases'].max() for i in model.predict(input_x)]

    d['fit'] = model.predict(input_x)
fig = go.Figure()



for d in dat:

  country_name, color_key, fit = itemgetter('name', 'color', 'fit')(d)

  country = countries[country_name]

  fig.add_trace(go.Scatter(

    x = country['Date'],

    y = country['ConfirmedCases'],

    name = country_name,

    line = {'color': colors[color_key],

            'width': 3}

    #linewidth=3

  ))

  fig.add_trace(go.Scatter(

    x = all_dates,

    y = [f[0] * countries[country_name]['ConfirmedCases'].max() for f in fit],

    name = country_name + ' (prediction)',

    line = {'color': colors[color_key],

            'dash' : 'dot'}

    #linestyle=':'

  ))



fig.update_layout(title_text='Confirmed Cases',

                  xaxis_rangeslider_visible=True)



fig.show()
fig = go.Figure()



for d in dat:

  country_name, color_key, fit = itemgetter('name', 'color', 'fit')(d)

  country = countries[country_name]

  fig.add_trace(go.Scatter(

    x = country['Date'],

    y = country['Fatalities'],

    name = country_name,

    line = {'color': colors[color_key],

            'width': 3}

    #linewidth=3

  ))

  fig.add_trace(go.Scatter(

    x = all_dates,

    y = [f[1] * countries[country_name]['Fatalities'].max() for f in fit],

    name = country_name + ' (prediction)',

    line = {'color': colors[color_key],

            'dash' : 'dot'}

    #linestyle=':'

  ))



fig.update_layout(title_text='Fatalities',

                  xaxis_rangeslider_visible=True)



fig.show()