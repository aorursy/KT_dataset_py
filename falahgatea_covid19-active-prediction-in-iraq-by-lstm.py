import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# import Dataset

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



data.info()
full_table = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['ObservationDate'])

full_table.head(-5)
# checking for missing value

full_table.isna().sum()

# cases for all world

cases = ['Confirmed', 'Deaths', 'Recovered']

# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

full_table['Country/Region'] = full_table['Country/Region'].replace('iraq', 'Iraq')

# iraq and the row

iraq_data = full_table[full_table['Country/Region']=='Iraq']
iraq_data
# deaths count in Iraq 

iraq_data['Deaths']
#Confirmed count in Iraq

iraq_data['Confirmed']
#Recovered count in iraq  

iraq_data['Recovered']

#Active now in iraq it is 

iraq_data['Active']
iraq_conf_cases = Iraq.groupby('ObservationDate').sum().apply(list).reset_index()

iraq_conf_cases

i=1

tot_conf = 0

#date_conf_cases['Total Confirmed Cases'] = 1

iraq_data['Days'] = 1

for ind in Iraq.index: 

    iraq_data['Days'][ind] = i

    i=i+1   
 #Confirmed & Deaths & Recovered &Active and Days for Iraq Now

iraq_data
#create dataframe only for days and active columns

x=iraq_data['Days']

y=iraq_data['Active']

raw_data = {'Days': x,'Active':y}

df_Iraq = pd.DataFrame(raw_data, columns = ['Days', 'Active'])

df_Iraq

#create two variable x and y for training dataset 

x=df_Iraq['Days']

y=df_Iraq['Active']



fig = px.line(iraq_data, x=x, y=y,title="Iraq Active Count Over Time")

fig.show()
#split dataset x,y

from sklearn.model_selection import train_test_split



x_train_active, x_test_active, y_train_active, y_test_active = train_test_split(x, y, test_size=0.1, shuffle=False)

print('train data: ', x_train_active.shape, y_train_active.shape)

print('validate data: ',x_test_active.shape, y_test_active.shape)

# and Now import model libriries from keras api

from keras import backend as K

from keras.models import Sequential, load_model, Model

from keras.layers import LSTM, Dense, Activation, Input, Reshape

from numpy import array

X = array(x).reshape(30, 1, 1)

# stacked LSTM Model

model = Sequential()

model.add(LSTM(50, activation='linear', return_sequences=True, input_shape=(1, 1)))

model.add(LSTM(50, activation='linear'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())

history=model.fit(X, y, epochs=300, validation_split=0.2, batch_size=5)

# plot train and validation loss

from matplotlib import pyplot

pyplot.plot(history.history['loss'])

pyplot.plot(history.history['val_loss'])

pyplot.title('model train vs validation loss')

pyplot.ylabel('loss')

pyplot.xlabel('epoch')

pyplot.legend(['train', 'validation'], loc='upper right')

pyplot.show()
#save model

model.save('iraq_active_lstm.h5')
#test new and Next Day 

x=31

print('Next Day:',x)

test_input = array([x])

print('Next Day after array :',test_input)

test_input = test_input.reshape((1, 1, 1))

print('test_input:',test_input)

test_output = model.predict(test_input, verbose=0).flatten()  

print('test_output:',test_output)

test_output =test_output.reshape(-1)

print('test_output:',test_output)

y=df_Iraq['Active']

y
y=np.array(y)

y
data=[]

for day in range(30):

    day=day+1

    day_input = array([day])

    day_input = day_input.reshape((1, 1, 1))

    active_output_prediced = model.predict(day_input, verbose=0).flatten()

    day_input=day_input.reshape(-1)

    active_output_prediced=active_output_prediced.reshape(-1)

    print('day_input:',day_input)

    print('Real:',y[day-1])

    print('active_output_prediced:', active_output_prediced)

    print('day:',day)

    raw_data = {'Days':day_input,'Active':np.array(y),'Active_Output_prediced':np.array(active_output_prediced)}

    data.append(raw_data)
data
x_value
y_prediction
y
x_value=df['Days']

y_prediction=y_prediction

y_real=y

plt.figure(figsize=(10,3))

plt.plot(y_real, color='blue', label='Actual')

plt.plot(y_prediction , color='red', label='Predicted ')

plt.title('Covid19 Active In Iraq Real & Prediction')

plt.xlabel('Day')

plt.ylabel('Acive Count')

plt.legend()

plt.savefig('Covide19_ active_in_Iraq_Real_Prediction.png')

plt.savefig('Covide19_ active_in_Iraq_Real_Prediction.pdf')

plt.show()

#save too keras model

model.save('Covide19_ active_in_Iraq.h5')