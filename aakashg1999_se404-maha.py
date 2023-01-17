import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

import matplotlib.pyplot as plt

import re

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import plotly.express as px



import numpy as np

import pandas as pd

import os

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 

from statsmodels.tsa.seasonal import seasonal_decompose 

#from pmdarima import auto_arima                        

from sklearn.metrics import mean_squared_error

from statsmodels.tools.eval_measures import rmse

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()



import plotly.express as px

import plotly.graph_objects as go



import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')





import statsmodels.api as sm



color = sns.color_palette()

sns.set_style('darkgrid')



from numpy.random import seed

seed(1)





import tensorflow

tensorflow.random.set_seed(1)
age_details = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

#india_covid_19 = pd.read_csv('covid_19_india.csv')

hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')

ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')

population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')



world_population = pd.read_csv('../input/covid19-in-india/population_india_census2011.csv')

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')



#india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'],dayfirst = True)

state_testing['Date'] = pd.to_datetime(state_testing['Date'])

ICMR_details['DateTime'] = pd.to_datetime(ICMR_details['DateTime'],dayfirst = True)

ICMR_details = ICMR_details.dropna(subset=['TotalSamplesTested', 'TotalPositiveCases'])
covid_19_india=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_19_india.Date = pd.to_datetime(covid_19_india.Date, dayfirst=True)
#Active cases of corona in India

covid_19_india['Active'] = covid_19_india['Confirmed'] - covid_19_india['Cured'] - covid_19_india['Deaths']

covid_19_india.tail()
state_per_day = covid_19_india.groupby(["Date" , "State/UnionTerritory"])["Confirmed","Deaths", "Cured" , "Active"].sum().reset_index().sort_values("Date", ascending = True)

kerala_per_day = state_per_day.loc[state_per_day['State/UnionTerritory'] == 'Kerala']

delhi_per_day = state_per_day.loc[state_per_day['State/UnionTerritory'] == 'Delhi']

maharshtra_per_day = state_per_day.loc[state_per_day['State/UnionTerritory'] == 'Maharashtra'] 

gujarat_per_day = state_per_day.loc[state_per_day['State/UnionTerritory'] == 'Gujarat']
df = maharshtra_per_day[['Date', 'Confirmed']]

df = df[df.Date >= '2020-03-27']

df.reset_index()

df = df.set_index("Date")

print(df)
dfa = maharshtra_per_day[['Date', 'Confirmed']]

dfa.days = dfa.index

dfa["Days"] = dfa.index[:]

dfa.drop(['Date'] , axis =1)
X_dfa = pd.DataFrame(dfa["Days"])

y_dfa = pd.DataFrame(dfa["Confirmed"])

X_traindfa = X_dfa[:-10]

y_traindfa = y_dfa[:-10]

X_testdfa = X_dfa[-10:]

y_testdfa = y_dfa[-10:]
y_testdfa.shape
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X_traindfa)
lr = LinearRegression()

lr.fit(X_poly ,y_traindfa)
y_trainpred = lr.predict(poly_reg.fit_transform(X_traindfa))

y_testpred = lr.predict(poly_reg.fit_transform(X_testdfa))
y_testpoly = y_dfa[-10:]

y_testpoly['Confirmed'] = y_testdfa[:]

y_testpoly['Predicted'] = y_testpred[:]

y_testpoly.round(1)
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_testpoly['Confirmed'], y_testpoly['Predicted']))

import math

print(math.sqrt(mean_squared_error(y_testpoly['Confirmed'], y_testpoly['Predicted'])))
plt.scatter(X_traindfa, y_traindfa, color = 'red')

plt.plot(X_traindfa, y_trainpred, color = 'blue')

plt.title(' cases of covid19(Polynomial Regression)')

plt.xlabel('days')

plt.ylabel('cases')

plt.show()
y_testdfa.shape
plt.scatter(X_testdfa, y_testdfa, color = 'red')

plt.plot(X_testdfa, y_testpred, color = 'blue')

plt.title(' cases of covid19(Polynomial Regression)')

plt.xlabel('days')

plt.ylabel('cases')

plt.show()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
n_size = 5
train_data = df[:len(df)-2*n_size]

validation_data = df[len(df)-2*n_size : len(df)-n_size]

test_data = df[len(df)-n_size:]

train_data.shape
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)

scaled_validation_data = scaler.transform(validation_data)

scaled_test_data = scaler.transform(test_data)
scaled_data = np.concatenate((scaled_train_data , scaled_validation_data, scaled_test_data) , 0)

scaled_data.shape
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
n_input = 20

scaled_temptest_data = scaled_data[-n_size-n_input:]

scaled_data.shape
scaled_data[-n_input-20:-n_input].shape

scaled_temptest_data.shape
from keras.backend import sigmoid

def swish(x, beta = 1):

    return (x * sigmoid(beta * x))
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Activation

get_custom_objects().update({'swish': Activation(swish)})
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers
n_input = 30
X_train = []

y_train = []

for i in range(n_input, len(scaled_train_data)):

    X_train.append(scaled_train_data[i-n_input:i, 0])

    y_train.append(scaled_train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)



# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = []

y_test = []

for i in range(len(scaled_data)-n_size , len(scaled_data)):

    X_test.append(scaled_data[i-n_input:i, 0])

    y_test.append(scaled_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)



X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_valid = []

y_valid = []

for i in range(len(scaled_data)-2*n_size , len(scaled_data)-n_size):

    X_valid.append(scaled_data[i-n_input:i, 0])

    y_valid.append(scaled_data[i, 0])

X_valid, y_valid = np.array(X_valid), np.array(y_valid)



X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
lstm_model = Sequential()

#lstm_model.add(LSTM(200, activation='tanh', input_shape=(n_input, n_features)))

lstm_model.add(LSTM(200,activation='swish' ,input_shape=(X_train.shape[1], 1)))

lstm_model.add(Dropout(0.5))

#lstm_model.add(LSTM(200, activation='relu', input_shape=(None)))

lstm_model.add(Dense(1))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

lstm_model.compile(optimizer='Adadelta', loss='mse', metrics=['accuracy'])#, learning_rate=0.01)



lstm_model.summary()
lstm_model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid , y_valid) , batch_size = 1 )
losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))

plt.xticks(np.arange(0,21,1))

plt.plot(range(len(losses_lstm)),losses_lstm);
plt.plot(lstm_model.history.history['loss'])

plt.plot(lstm_model.history.history['val_loss'])

plt.title('model train vs validation loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
predValid_case = lstm_model.predict(X_valid)

predTest_case = lstm_model.predict(X_test)

predTrain_case = lstm_model.predict(X_train)
predValid_casex = scaler.inverse_transform(predValid_case)

predTest_casex = scaler.inverse_transform(predTest_case)

predTrain_casex = scaler.inverse_transform(predTrain_case)
test_datatmp= test_data

test_datatmp["Predicted"] = predTest_casex[:]

test_datatmp.round(1)
valid_datatmp= validation_data

valid_datatmp["Predicted"] = predValid_casex[:]

valid_datatmp.round(1)
totest_data = pd.concat(( valid_datatmp , test_datatmp) , 0)

totest_data.round(1)
from sklearn.metrics import mean_squared_error

print(mean_squared_error(totest_data['Confirmed'], totest_data['Predicted']))

import math

print(math.sqrt(mean_squared_error(totest_data['Confirmed'], totest_data['Predicted'])))
train_datatmp= train_data[n_input:]

train_datatmp["Predicted"] = predTrain_casex[:]

train_datatmp.round(1)
train_datatmp['Confirmed'].plot(figsize = (16,5), legend=True)

train_datatmp['Predicted'].plot(legend = True)
valid_datatmp['Confirmed'].plot(figsize = (16,5), legend=True)

valid_datatmp['Predicted'].plot(legend = True)
test_datatmp['Confirmed'].plot(figsize = (16,5), legend=True)

test_datatmp['Predicted'].plot(legend = True)
totest_data['Confirmed'].plot(figsize = (16,5), legend=True)

totest_data['Predicted'].plot(legend = True)