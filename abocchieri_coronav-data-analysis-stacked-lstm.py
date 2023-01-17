import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import time

import datetime as dt

from datetime import date

import warnings

warnings.filterwarnings('ignore')
covdata_filepath = '/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv'

covdata = pd.read_csv(covdata_filepath)

#covdata.head()

covdata['Last Update'] = covdata['Last Update'].apply(pd.to_datetime)

covdata.drop(['SNo'],axis=1,inplace=True)

covdata.tail()
countries = covdata['Country/Region'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
covdata_china = covdata.loc[covdata['Country/Region'] == 'Mainland China']

covdata_china.tail()
Mainland_China_provinces = covdata_china['Province/State'].unique().tolist()

print(Mainland_China_provinces)

cases = 0

for province in Mainland_China_provinces:

    province = covdata_china.loc[covdata_china['Province/State'] == province]

    cases += province['Confirmed'].iloc[[-1][0]]

print('Mainland China total cases number is : ', int(cases))



hubei = covdata_china.loc[covdata_china['Province/State'] == 'Hubei']

#hubei.tail()

hubei_cases = hubei['Confirmed'].iloc[[-1][0]]

print('Hubei confirmed cases: ', int(hubei_cases))



print('Mainland China cases w/ Hubei: ', int(cases - hubei_cases))
df_hubei = pd.DataFrame(hubei)

dates = df_hubei.ObservationDate

x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]



f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.plot(x, df_hubei.Confirmed, '.y')

ax.plot(x, df_hubei.Deaths, '.r')

ax.plot(x, df_hubei.Recovered, '.g')

plt.gcf().autofmt_xdate()

ax.xaxis.set_major_locator(ticker.AutoLocator())

plt.show()
covdata_US = covdata.loc[covdata['Country/Region'] == 'US']

US_states = covdata_US['Province/State'].unique().tolist()

#print(US_states)

cases = 0

cases_max = cases

for state in US_states:

    state = covdata_US.loc[covdata_US['Province/State'] == state]

    cases_state = state['Confirmed'].iloc[[-1][0]]

    cases += cases_state

    if cases_state > cases_max:

        cases_max = cases_state

        state_with_max_number = state['Province/State'].iloc[-1]

        #print('in {} there are {} cases.'.format(state['Province/State'].iloc[-1], cases_max))

print('US total cases number is: ', int(cases))

print('The US state/province with the max number of cases is: {}, having {} cases'.format(state_with_max_number, int(cases_max)))

NY = covdata_US.loc[covdata_US['Province/State'] == 'New York']
df_NY = pd.DataFrame(NY)

dates = df_NY.ObservationDate

x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]



f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.plot(x, df_NY.Confirmed, '.y')

ax.plot(x, df_NY.Deaths, '.r')

ax.plot(x, df_NY.Recovered, '.g')

plt.gcf().autofmt_xdate()

ax.xaxis.set_major_locator(ticker.AutoLocator())

plt.show()
covdata_new = covdata[covdata['Last Update'] > pd.Timestamp(date(2020,2,15))]

covdata_new.head()
df = pd.DataFrame(covdata_new)

df = df.loc[df['Country/Region'] == 'Italy']

df.tail()
dates = df.ObservationDate

x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates]



f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.plot(x, df.Confirmed, '.y')

ax.plot(x, df.Deaths, '.r')

ax.plot(x, df.Recovered, '.g')

plt.gcf().autofmt_xdate()

ax.xaxis.set_major_locator(ticker.AutoLocator())

plt.show()
f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.set_yscale('log')

ax.plot(x, df.Confirmed, '.y')

ax.plot(x, df.Deaths, '.r')

ax.plot(x, df.Recovered, '.g')

plt.gcf().autofmt_xdate()

ax.xaxis.set_major_locator(ticker.AutoLocator())

plt.show()
new_Confirmed = [0]

new_Deaths = [0]

new_Recovered = [0]

for i in range(1, (df.Confirmed).size):

    new_Confirmed.append(df.Confirmed.tolist()[i]-df.Confirmed.tolist()[i-1])

    new_Deaths.append(df.Deaths.tolist()[i]-df.Deaths.tolist()[i-1])

    new_Recovered.append(df.Recovered.tolist()[i]-df.Recovered.tolist()[i-1])
f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.set_xscale('log')

ax.set_yscale('log')

ax.plot(df.Confirmed, new_Confirmed, '.y', label = 'italian confirmed')

ax.plot(df.Deaths, new_Deaths,  '.r', label = 'italian dead')

ax.plot(df.Recovered, new_Recovered,  '.g', label = 'italian recovered')



ax.legend(loc = 'best')

plt.show()
y = np.log10(new_Confirmed) ; cleany = []; cleanx = []

i = 0

for j in y:

    if j != -np.inf: 

        cleany.append(j); 

        cleanx.append(np.log10(df.Confirmed.tolist()[i]))

    i +=1    

x_confirmed = np.asarray(cleanx); y = np.asarray(cleany)

a_confirmed, b_confirmed = np.polyfit(x_confirmed, y, 1)



y = np.log10(new_Deaths) ; cleany = []; cleanx = []

i = 0

for j in y:

    if j != -np.inf: 

        cleany.append(j); 

        cleanx.append(np.log10(df.Deaths.tolist()[i]))

    i +=1

x_deaths = np.asarray(cleanx); y = np.asarray(cleany)

a_deaths, b_deaths = np.polyfit(x_deaths, y, 1)



y = np.log10(new_Recovered) ; cleany = []; cleanx = []

i = 0

for j in y:

    if j != -np.inf:

        if np.isnan(j) != True:

            if df.Recovered.tolist()[i] != 0:

                cleany.append(j)        

                cleanx.append(np.log10(df.Recovered.tolist()[i]))

    i +=1

x_recovered = np.asarray(cleanx); y = np.asarray(cleany)

a_recovered, b_recovered = np.polyfit(x_recovered, y, 1)
f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.set_xscale('log')

ax.set_yscale('log')

ax.plot(df.Confirmed, new_Confirmed, '.y')

ax.plot(10**(x_confirmed), 10**(a_confirmed*x_confirmed + b_confirmed), 'y', label = 'italian confirmed')

ax.plot(df.Deaths, new_Deaths,  '.r')

ax.plot(10**(x_deaths), 10**(a_deaths*x_deaths + b_deaths), 'r', label = 'italian dead')

ax.plot(df.Recovered, new_Recovered,  '.g')

ax.plot(10**(x_recovered), 10**(a_recovered*x_recovered + b_recovered), 'g', label = 'italian recovered')



ax.legend(loc = 'best')

plt.show()
new_cases_hubei = [0]; new_cases_NY = [0]

for i in range(1, (df_hubei.Confirmed).size):

    new_cases_hubei.append(df_hubei.Confirmed.tolist()[i]-df_hubei.Confirmed.tolist()[i-1])

for i in range(1, (df_NY.Confirmed).size):

    new_cases_NY.append(df_NY.Confirmed.tolist()[i]-df_NY.Confirmed.tolist()[i-1])
y = np.log10(new_cases_NY) ; cleany = []; cleanx = []

i = 0

for j in y:

    if j != -np.inf: 

        cleany.append(j); 

        cleanx.append(np.log10(df_NY.Confirmed.tolist()[i]))

    i +=1    

x_confirmed_NY = np.asarray(cleanx); y = np.asarray(cleany)

a_confirmed_NY, b_confirmed_NY = np.polyfit(x_confirmed_NY, y, 1)
f, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.set_xscale('log')

ax.set_yscale('log')

ax.plot(df.Confirmed, new_Confirmed, '.y')

ax.plot(10**(x_confirmed), 10**(a_confirmed*x_confirmed + b_confirmed), 'y', label = 'italian confirmed')



ax.plot(df_hubei.Confirmed, new_cases_hubei, '.k', label = 'Hubei_cases')

ax.plot(df_NY.Confirmed, new_cases_NY, '.b')

ax.plot(10**(x_confirmed_NY), 10**(a_confirmed_NY*x_confirmed_NY + b_confirmed_NY), 'b', label = 'NY confirmed')



ax.legend(loc = 'best')

plt.gcf().autofmt_xdate()

plt.show()
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout
# split a univariate sequence into samples

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
raw_seq = np.array(new_Confirmed)

n_steps = 5

X, y = split_sequence(raw_seq, n_steps)



# summarize the data

#for i in range(len(X)):

	#print(X[i], y[i])

    

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model

model1 = Sequential()

model1.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

model1.add(Dropout(0.1))

model1.add(LSTM(50, activation='relu', return_sequences=True))

model1.add(Dropout(0.1))

model1.add(LSTM(50, activation='relu'))

model1.add(Dense(1))



start = time.time()

model1.compile(optimizer='adam', loss='mse')          

print('compilation time : ', time.time() - start)

print('\n')

model1.summary()
model1.fit(X, y, epochs=300, verbose=0)

x_input = np.array(new_Confirmed)[-n_steps:]

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model1.predict(x_input, verbose=0)

cases_forecast = int(round(yhat[0][0]))

print('cases forecast for today: {}'.format(cases_forecast))



accuracy = 100*(1-np.abs(cases_forecast - new_Confirmed[-1])/new_Confirmed[-1])

print('forecast accuracy: {:.2f} %'.format(accuracy))



print('and for tomorrow?')

x_input2 = np.array(new_Confirmed)[(-n_steps+1):]

x_input2 = np.concatenate((x_input2, yhat), axis=None)

x_input2 = x_input2.reshape((1, n_steps, n_features))

yhat2 = model1.predict(x_input2, verbose=0)

print('tomorrow there might be {:.0f} cases'.format(int(yhat2[0][0])))
raw_seq = np.array(new_Deaths)

n_steps = 4

X, y = split_sequence(raw_seq, n_steps)



n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))



# define model

model2 = Sequential()

model2.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

model2.add(LSTM(50, activation='relu', return_sequences=True))

model2.add(LSTM(50, activation='relu'))

model2.add(Dense(1))



start = time.time()

model2.compile(optimizer='adam', loss='mse')          

print('compilation time : ', time.time() - start)

print('\n')

model2.summary()
model2.fit(X, y, epochs=300, verbose=0)



x_input = np.array(new_Deaths)[-n_steps:]

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model2.predict(x_input, verbose=0)



deaths_forecast = int(round(yhat[0][0]))

print('deaths forecast for today: {}'.format(deaths_forecast))



accuracy = 100*(1-np.abs(deaths_forecast - new_Deaths[-1])/new_Deaths[-1])

print('forecast accuracy: {:.2f} %'.format(accuracy))



print('and for tomorrow?')

x_input2 = np.array(new_Deaths)[(-n_steps+1):]

x_input2 = np.concatenate((x_input2, yhat), axis=None)

x_input2 = x_input2.reshape((1, n_steps, n_features))

yhat2 = model2.predict(x_input2, verbose=0)

print('tomorrow, in total, there might be {:.0f} deaths'.format(int(yhat2[0][0])))