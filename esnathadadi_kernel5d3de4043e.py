import tensorflow as tf
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt # this is used for the plot the graph 

import sklearn

import seaborn as sns # used for plot interactive graph

sns.set(style="darkgrid")
from numpy import array
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense #Dropout is to avoid overfitting

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.layers import Bidirectional

from keras.layers import TimeDistributed

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split  # to split the data into two parts

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics # for the check the error and accuracy of the model

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate

from keras.layers import Bidirectional
import pandas as pd

import numpy as np

import re

import datetime

import pickle

from sklearn.externals import joblib



import matplotlib.pyplot as plt
data = pd.read_csv('C:\\Users\\Ali\\Documents\\datasets\\covid_19_data.csv')
data.head()
data.describe()
data.shape
corr_matrix = data.corr()

print(corr_matrix)
sns.heatmap(corr_matrix)
data.dropna()
#create a new column called active

data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
print(data['Active'])
data.tail()
#drop the columns Sno and Last Update

data.drop("SNo", axis=1, inplace=True)

data.drop("Last Update", axis=1, inplace=True)
data.hist(bins=50, figsize=(20,15))

plt.show()

#list of countries with confirmed cases

country_list = list(data['Country/Region'].unique())

print(country_list)
#list of countries infected with COVID-19

print(len(country_list))
#total number of cases per country

byCountry=data.groupby(['Country/Region']).max().reset_index(drop=None)

print(byCountry[['Country/Region', 'Confirmed','Deaths','Recovered']])
#converting ObservationDate to date object

data['ObservationDate'] = pd.to_datetime(data['ObservationDate'], format='%m/%d/%Y')

print(data['ObservationDate'])

#preparing for a timeseries analysis

data_by_date = data.groupby(['ObservationDate']).sum().reset_index(drop=None)

data_by_date['daily_cases'] = data_by_date.Confirmed.diff()

data_by_date['daily_recoveries'] = data_by_date.Recovered.diff()

data_by_date['daily_deaths'] = data_by_date.Deaths.diff()

print(data_by_date)
df = data.loc[:,'Confirmed':'Recovered']

print(df)
#normalise the dataset

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

df = scaler.fit_transform(df)

df.shape
y = df[:, 1]

print(y)
x = df

print(x)
#splitting the dataframe into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# convert a dataset of values into an array  matrix

def create_dataset(dataset,look_back=1):

    dX,dY=[],[]

    for i in range(len(df)-look_back-1):

        a=df[i:(i+look_back),0]

        dX.append(a)

        dY.append(df[i+look_back,0])

    return np.array(dX),np.array(dY)
#reshaping data into 3D

trainX=np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))

testX=np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))
#Creating an lstm

model = Sequential()

model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(1, 3)))

model.add(LSTM(64, activation='relu', return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(50, activation='relu', return_sequences=True))

model.add(LSTM(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1))



#compile model

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

print(model.summary())
history = model.fit(trainX, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)
# learning curves of model accuracy

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()
filename='model.pkl'

joblib.dump(model, filename)
#import library for plotting 3D diagrams

from mpl_toolkits.mplot3d import Axes3D
sns.set(color_codes=True)
sns.regplot(x="Confirmed", y="Deaths", data=data)
sns.regplot(x="Confirmed", y="Recovered", data=data)
#fig = plt.figure()

#ax = Axes3D(fig)

#plt.plot(data['Confirmed'], data['Recovered'], data['Deaths'])

#plt.show()
#rate of death versus rate of recovery

plt.plot('ObservationDate','Deaths', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='red')

plt.plot('ObservationDate','Recovered', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='green')

plt.xticks(rotation=60)

plt.ylabel('Number of cases', fontsize=15)

plt.xlabel('Dates', fontsize=15)

plt.legend()

plt.show()
#rate of death versus rate of recovery

plt.plot('ObservationDate','Deaths', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='red')

plt.plot('ObservationDate','Recovered', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='green')

plt.plot('ObservationDate','Active', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='yellow')

plt.xticks(rotation=60)

plt.title('Comparison of Active cases, Recovered cases and Deaths due to Covid-19')

plt.ylabel('Number of cases', fontsize=15)

plt.xlabel('Dates', fontsize=15)

plt.legend()

plt.show()
#Ten countries with the highest prevalence

plt.rcParams['figure.figsize']=(15,7)



sns.barplot(

x='Country/Region',

y='Confirmed',

data=byCountry[byCountry['Country/Region'] != 'Mainland China'].nlargest(10, 'Confirmed'), palette=sns.cubehelix_palette(15,reverse=True))

#Ten countries with the highest recoveries

plt.rcParams['figure.figsize']=(15,7)



sns.barplot(

x='Country/Region',

y='Recovered',

data=byCountry[byCountry['Country/Region'] != 'Mainland China'].nlargest(10, 'Recovered'), palette=sns.cubehelix_palette(15,reverse=True))
#Ten countries with the highest deathrate

plt.rcParams['figure.figsize']=(15,7)



sns.barplot(

x='Country/Region',

y='Deaths',

data=byCountry[byCountry['Country/Region'] != 'Mainland China'].nlargest(10, 'Deaths'), palette=sns.cubehelix_palette(15,reverse=True))
#Ten countries with the least prevalence

plt.rcParams['figure.figsize']=(15,7)



sns.barplot(

x='Country/Region',

y='Confirmed',

data=byCountry[byCountry['Country/Region'] != 'Mainland China'].nsmallest(10, 'Confirmed'), palette=sns.cubehelix_palette(15,reverse=True))
#Ten countries with the least deathrate

plt.rcParams['figure.figsize']=(15,7)



sns.barplot(

x='Country/Region',

y='Deaths',

data=byCountry[byCountry['Country/Region'] != 'Mainland China'].nsmallest(10, 'Deaths'), palette=sns.cubehelix_palette(15,reverse=True))
#visual comparison of the number of confirmed, recovered, active cases and deaths.

X = len(data)

plt.bar(X + 0.00, data['Confirmed'], color = 'b', width = 0.25)

plt.bar(X + 0.25, data['Active'], color = 'm', width = 0.25)

plt.bar(X + 0.50, data['Recovered'], color = 'g', width = 0.25)

plt.bar(X + 0.75, data['Deaths'], color = 'r', width = 0.25)

plt.title('Visual presentation of confirmed cases against deaths, active cases and recovered cases', size=20)

plt.xlabel('Condition')

plt.ylabel('Number of cases')

plt.show()
#A = data['Confirmed']

#B = data['Deaths']

#X = len(data)

#plt.bar(X, A, color = 'b')

#plt.bar(X, B, color = 'r', bottom = A)

#plt.show()
#graph showing trend of Corona Virus over time

#X = data['ObservationDate']

#Y = data['Confirmed']

#plt.plot(X, Y, c = 'k')

plt.plot('ObservationDate','Confirmed', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='red')

plt.title('Disease prevalence over time', size=16)

plt.grid(True, lw = 2, ls = '--', c = '.75')

plt.show()
#graph showing trend of Deaths due to Corona Virus over time

#X = data['ObservationDate']

#Y = data['Deaths']

#plt.plot(X, Y, c = 'r')

plt.plot('ObservationDate','Deaths', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='magenta')

plt.title('Progression of deaths over time', size=16)

plt.xlabel('Dates')

plt.ylabel('Number of deaths')

plt.grid(True, lw = 2, ls = '--', c = '.75')

plt.show()
#graph showing trend of recovered patients who were diagnosed with Corona Virus over time

#X = data['ObservationDate']

#Y = data['Recovered']

#plt.plot(X, Y, c = 'm')

plt.plot('ObservationDate','Recovered', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='grey')

plt.xlabel('Observation Date')

plt.ylabel('Number of recovered cases')

plt.grid(True, lw = 1, ls = '--', c = '1')

plt.show()
plt.plot('ObservationDate','Active', data=data.groupby(['ObservationDate']).sum().reset_index(drop=None),color='green')

plt.xlabel('observation Date', fontsize=13)

plt.ylabel('Active Cases', fontsize=13)

plt.title('Active Covid-19 cases over time', fontsize=17)
#confirmed cases by country

#plt.figure(figsize=(32, 18))

#plt.barh(data['Confirmed'], data['Country/Region'])

#plt.title('Number of Coronavirus Confirmed Cases in Countries/Regions', size=20)

#plt.show()
#a graph showing the number of countries with confirmed cases of COVID-19

data['Country/Region'].value_counts().hist()
#graphs showing distribution of the dataset

#data.hist(bins=20, figsize=(20,15))

#plt.show()
Italy = data[data['Country/Region']=='Italy']

print(Italy)
data.head()
data[data['Country/Region']=='Italy'].sum()
#plot graph for deaths in Italy

Italy['Deaths'].plot()
data[data['Country/Region']=='Zimbabwe'].sum()
print(data[data['Country/Region']=='Zimbabwe'])
Zimbabwe = data[data['Country/Region']=='Zimbabwe']

print(Zimbabwe)
Zimbabwe['Confirmed'].plot()
Zimbabwe['Deaths'].plot()
#patients who recovered from COVID-19 in Zimbabwe

Zimbabwe['Recovered'].plot()

plt.xlabel('Total number of COVID-19 patients worldwide')

plt.ylabel('Number of recovered cases in Zimbabwe')
#active COVID=19 cases

data['Active'].plot()

plt.title('Active COVID-19 cases')

plt.xlabel('total confirmed cases worldwide')

plt.ylabel('Number of currently active cases')
#visualisation of deaths progression worldwide

data_by_date['daily_deaths'].plot()
#visualisation of COVID-19 prevalence worldwide

data_by_date['daily_cases'].plot()
#visualisation of recovered cases progression worldwide

data_by_date['daily_recoveries'].plot()