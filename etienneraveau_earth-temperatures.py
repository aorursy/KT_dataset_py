# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')
print(temperatures.columns)
print(temperatures.head)
# Start with some line plots of temperatures' evolution over time
import matplotlib.pyplot as plt

# Land average temperature
plt.figure(figsize=(15,15))
plt.plot(temperatures['dt'],temperatures['LandAverageTemperature'])
plt.plot(temperatures['dt'],temperatures['LandAverageTemperatureUncertainty'])
plt.legend(['Average temperature','Average temperature uncertainty'])
plt.title('Land average temperature evolution over almost 300 years')
plt.xlabel('Date time')
plt.ylabel('Average Temperature (celsius degrees)')
plt.show()

# One can observe here that average temperature is globally slightly increasing but its variance remains quite unchanged 
# throughout the years. In addition to this last observation, the slightly greater variance in first recorded years seems
# explained by an increased uncertainty over measures of temperatures at that time. Likewise, one can argue that this
# observation might reinforce the first hint which is an increasing average temperature as for lands over the years.
# Land and Oceans average temperature
plt.figure(figsize=(15,15))
plt.plot(temperatures['dt'],temperatures['LandAndOceanAverageTemperature'])
plt.legend(['Average temperature'])
plt.title('Lands and Oceans average temperature evolution over almost 300 years')
plt.xlabel('Date time')
plt.ylabel('Average Temperature (celsius degrees)')
plt.show()

# The previous observation are reinforced by this more general chart.
# One should note that there is fewer data here : this chart essentially contains data with datetimes associated to low
# uncertainty on the previous graph.
# Let's take a closer look at France's situation right now.
temperatures_country = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
#print(temperatures_country.head)

# Get list of countries having recorded data
#print(np.unique(temperatures_country['Country']))

# Print France's data and France (Europe)'s
plt.figure(figsize=(15,15))

plt.subplot(121)
plt.plot(temperatures_country['dt'][temperatures_country['Country']=='France'],temperatures_country['AverageTemperature'][temperatures_country['Country']=='France'],color='c')
plt.legend(['Average France\'s temperature'])
plt.title('France\'s average temperature evolution over almost 300 years')
plt.xlabel('Date time')
plt.ylabel('Average Temperature (celsius degrees)')

plt.subplot(122)
plt.plot(temperatures_country['dt'][temperatures_country['Country']=='France (Europe)'],temperatures_country['AverageTemperature'][temperatures_country['Country']=='France (Europe)'],color='m')
plt.legend(['Average France (Europe)\'s temperature'])
plt.title('France (Europe)\'s average temperature evolution over almost 300 years')
plt.xlabel('Date time')
plt.ylabel('Average Temperature (celsius degrees)')

plt.show()
# Finnally, this cell is not useful as px.choropleth has been used in next cell rather than px.choropleth_mapbox
# Load json giving the countries boundaries as polygons
import json
with open('/kaggle/input/countriesgeo/countriesgeo.json') as response:
    counties = json.load(response)
    
print(np.unique(temperatures_country['Country']))

# Get all countries considered in the geojson file
to_keep = []
for i in range(len(counties['features'])):
    to_keep.append(counties['features'][i]['properties']['name'])

# Consider new dataframe with data only for to_keep refered countries
temperatures_country_tokeep = temperatures_country.loc[temperatures_country['Country'].isin(to_keep)]
# Let's plot the evolution of countries temperature over time in a choropleth map
import plotly.express as px

fig = px.choropleth(temperatures_country, 
                    locations="Country", 
                    locationmode = "country names",
                    color="AverageTemperature", 
                    hover_name="Country", 
                    animation_frame="dt")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
from sklearn.preprocessing import MinMaxScaler

# Create training and testing sets
ratio = 0.8
input_length = 12*3         # Here we take 3 years as input length
output_length = 1

# Eliminate na's from temperatures['LandAndOceanAverage']
data_lstm = temperatures['LandAndOceanAverageTemperature'].dropna()
data_lstm.reset_index(inplace=True,drop=True)
data_lstm = data_lstm.values

# Scale data between 0 and 1
scaler = MinMaxScaler()
data_lstm = scaler.fit_transform(np.reshape(data_lstm,(-1,1)))

# Split between training and testing sets
split = (int)(np.ceil(ratio*len(data_lstm)))
x_train = [data_lstm[i:i+input_length] for i in range(split-input_length)]
y_train = [data_lstm[i+input_length][0] for i in range(split-input_length)]
x_test = [data_lstm[i+split:i+split+input_length] for i in range(len(data_lstm)-split-input_length)]
y_test = [data_lstm[i+split+input_length][0] for i in range(len(data_lstm)-split-input_length)]

# Check shapes and look at some of the values
print(np.shape(x_train),np.shape(y_train))
print(np.shape(x_test),np.shape(y_test))
print(x_train[0])
print(y_train[0])
print(x_test[0])
print(y_test[0])

# Reshape x_train and x_test in order to be used in LSTM layers
x_train_lstm = np.reshape(x_train, (np.shape(x_train)[0], np.shape(x_train)[1], 1))
x_test_lstm = np.reshape(x_test, (np.shape(x_test)[0], np.shape(x_test)[1], 1))

print(np.shape(x_train_lstm))
# Try to predict Average Temperature evolution over next decades regarding land and ocean temperatures

# Import deep learning libraries
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

# Build our model
lstm = Sequential()
 
# Declare the layers
layers = [LSTM(units=128, input_shape=(input_length,1), activation='relu',return_sequences=True),
          LSTM(units=128, activation='relu',return_sequences=True),
          LSTM(units=128, activation='relu'),
         Dense(output_length)]
 
# Add the layers to the model
for layer in layers:
    lstm.add(layer)

# Compile our model
lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
 
# Fit the model
history_lstm = lstm.fit(x_train_lstm, y_train, validation_data=(x_test_lstm,y_test), epochs=10, batch_size=32)
# Plot loss evolution over training
plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')  
plt.xlabel('epochs')
plt.legend(['train','val'], loc='upper left')
# Make some predictions over 50 years and plot them
x_pred = [data_lstm[i:i+input_length] for i in range(len(data_lstm)-input_length)]
x_pred = np.reshape(x_pred,(np.shape(x_pred)[0],np.shape(x_pred)[1],1))

# Make prediction
pred = lstm.predict(x_pred)

# Plot results
plt.figure(figsize=(10,10))
plt.plot([i for i in range(len(data_lstm))],scaler.inverse_transform(data_lstm),color='b')
plt.plot([i for i in range(len(data_lstm),len(data_lstm)+50)],scaler.inverse_transform(pred[-50:]),color='r')
plt.legend(['recorded data','prediction'])
plt.xlabel('time units')
plt.ylabel('Average temperature (celsius degree)')
plt.title('Average land and ocean temperature forecast over next 50 years.')
plt.show()