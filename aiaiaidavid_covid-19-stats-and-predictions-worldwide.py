import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn')
              
# Set precision to two decimals
pd.set_option("display.precision", 2)

# Define date format for charts like Apr 16 or Mar 8
my_date_fmt = mdates.DateFormatter('%b %e')
# Download files from github
cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df_cases = pd.read_csv(cases_url, error_bad_lines=False)

deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df_deaths = pd.read_csv(deaths_url, error_bad_lines=False)

recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
df_recovered = pd.read_csv(recovered_url, error_bad_lines=False)
# Drop Province/State, Lat and Long
df_cases.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_deaths.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_recovered.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

# Rename Country/Region as Country
df_cases.rename(columns={'Country/Region' : 'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region' : 'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region' : 'Country'}, inplace=True)

# Some countries (Australia, Canada...) report data by province so we need to aggregate it
df_cases = df_cases.groupby(by='Country').sum()
df_deaths = df_deaths.groupby(by='Country').sum()
df_recovered = df_recovered.groupby(by='Country').sum()

# Transpose dataframes and make the date column index of datetime type
df_cases = df_cases.T
df_cases.index = pd.to_datetime(df_cases.index)
df_deaths = df_deaths.T
df_deaths.index = pd.to_datetime(df_deaths.index)
df_recovered = df_recovered.T
df_recovered.index = pd.to_datetime(df_recovered.index)
# Get last date in the set
last_date = df_cases.tail(1).index[0]
print('Last date in the set: ' + str(datetime.date(last_date)))
# List of countries for this work
country_list = ['Belgium', 'France', 'Germany', 'Italy', 'Netherlands', 'Portugal', 'Spain', 'Sweden', 'Switzerland',  
                 'Brazil', 'Canada', 'China', 'India', 'Iran',  'Mexico', 'Russia', 'United Kingdom', 'US']
clist1 = ['Belgium', 'France', 'Germany', 'Italy', 'Netherlands', 'Portugal', 'Spain', 'Sweden', 'Switzerland']
clist2 = ['Brazil', 'Canada', 'China', 'India', 'Iran', 'Mexico', 'Russia', 'United Kingdom', 'US']
# Extract selection of countries
df_cases = df_cases[country_list]
df_recovered = df_recovered[country_list]
df_deaths = df_deaths[country_list]
# Active cases = Confirmed cases - Recoverres - Deaths
df_active = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
for x in country_list:
    df_active[x] = df_cases[x] - df_recovered[x] - df_deaths[x] 
# Mortality(%) = Deaths / Cases
df_mortality = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
for x in country_list:
    df_mortality[x] = 100 * df_deaths[x] / df_cases[x] 
# Compute daily variation of confirmed and active cases, and deaths
df_cases_diff = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
df_active_diff = pd.DataFrame(columns=df_active.columns, index=df_active.index)
df_deaths_diff = pd.DataFrame(columns=df_deaths.columns, index=df_deaths.index)

for x in country_list:
    df_cases_diff[x] = df_cases[x].diff()
    df_active_diff[x] = df_active[x].diff()
    df_deaths_diff[x] = df_deaths[x].diff()
    
df_cases_diff.fillna(value=0, inplace=True)
df_active_diff.fillna(value=0, inplace=True)
df_deaths_diff.fillna(value=0, inplace=True)
# Confirmed cases and deaths are always growing, hence their derivatives must be positive or zero
df_cases_diff[df_cases_diff < 0] = 0
df_deaths_diff[df_deaths_diff < 0] = 0
# First batch of 9 countries: EVOLUTION of CASES (1 of 2)
fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0
for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(21)) 
  ax1[i,j].plot(df_cases.index, df_cases[x], color='navy', linewidth=2, label='Confirmed cases')
  ax1[i,j].plot(df_active.index, df_active[x], color='skyblue', linewidth=2, label='Active cases')
  ax1[i,j].plot(df_recovered.index, df_recovered[x], color='lime', linewidth=2, label='Recovered cases')
  ax1[i,j].plot(df_deaths.index, df_deaths[x], color='coral', linewidth=2, label='Deaths')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='upper left', fontsize='large')
fig1.suptitle('Evolution of covid-19 cases by country (Europe)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Second batch of 9 countries: EVOLUTION of CASES (2 of 2)
fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0
for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(21)) 
  ax2[i,j].plot(df_cases.index, df_cases[x], color='navy', linewidth=2, label='Confirmed cases')
  ax2[i,j].plot(df_active.index, df_active[x], color='skyblue', linewidth=2, label='Active cases')
  ax2[i,j].plot(df_recovered.index, df_recovered[x], color='lime', linewidth=2, label='Recovered cases')
  ax2[i,j].plot(df_deaths.index, df_deaths[x], color='coral', linewidth=2, label='Deaths')  
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax2[0,0].legend(loc='upper left', fontsize='large')
fig2.suptitle('Evolution of covid-19 cases by country (World excluding Europe)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Mortality(%) of covid-19 in Europe
print('Mortality(%) in Europe')
df_mortality[clist1].tail(1)
# Mortality(%) of covid-19 in the world (excl. Europe)
print('Mortality(%) in the world (excl.Europe)')
df_mortality[clist2].tail(1)
# First batch of 9 countries: DAILY VARIATION of CONFIRMED CASES (1 of 2)

fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax1[i,j].bar(df_cases_diff.index,  df_cases_diff[x], color='grey', alpha=0.2, label='Cases growth rate')
  ax1[i,j].plot(df_cases_diff.index,  df_cases_diff[x].rolling(window=7).mean(), color='indigo', linewidth=1.5, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='upper left', fontsize='large')
fig1.suptitle('Daily variation of covid-19 confirmed cases by country (Europe)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Second batch of 9 countries: DAILY VARIATION of CONFIRMED CASES (2 of 2)

fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax2[i,j].bar(df_cases_diff.index,  df_cases_diff[x], color='grey', alpha=0.2, label='Cases growth rate')
  ax2[i,j].plot(df_cases_diff.index,  df_cases_diff[x].rolling(window=7).mean(), color='navy', linewidth=1.5, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax2[0,0].legend(loc='upper left', fontsize='large')
fig2.suptitle('Daily variation of covid-19 confirmed cases by country (World excluding Europe)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()
# First batch of 9 countries: DAILY VARIATION of DEATHS (1 of 2)

fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax1[i,j].bar(df_deaths_diff.index,  df_deaths_diff[x], color='grey', alpha=0.2, label='Deaths growth rate')
  ax1[i,j].plot(df_deaths_diff.index,  df_deaths_diff[x].rolling(window=7).mean(), color='coral', linewidth=2, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='upper left', fontsize='large')
fig1.suptitle('Daily variation of covid-19 deaths by country (Europe)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Second batch of 9 countries : DAILY VARIATION of DEATHS (2 of 2)

fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))  
  ax2[i,j].bar(df_deaths_diff.index,  df_deaths_diff[x], color='grey', alpha=0.2, label='Deaths growth rate')
  ax2[i,j].plot(df_deaths.index,  df_deaths_diff[x].rolling(window=7).mean(), color='coral', linewidth=2, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax2[0,0].legend(loc='upper left', fontsize='large')
fig2.suptitle('Daily variation of covid-19 deaths by country (World excluding Europe)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Countries populations
# Source: https://www.worldometers.info/world-population/population-by-country/

pop = {}

pop['Belgium'] = 11589623
pop['France'] = 65273511
pop['Germany'] = 83783942
pop['Italy'] = 60461826
pop['Netherlands'] = 17134872
pop['Portugal'] = 10196709
pop['Spain'] = 46754778
pop['Sweden'] = 10099265
pop['Switzerland'] = 8654622
pop['Brazil'] = 212559417
pop['Canada'] = 37600000
pop['China'] = 1439323776
pop['India'] = 1380004385
pop['Iran'] = 83992949
pop['Mexico'] = 128932753
pop['Russia'] = 145934462
pop['United Kingdom'] = 67886011
pop['US'] = 331002651

pop
# Calculate nbr of cases per million people
df_cases_per_million = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
for x in df_cases_per_million.columns:
  df_cases_per_million[x] = 1000000 * df_cases[x] // pop[x]

print('Nbr of covid-19 cases per million people')
df_cases_per_million.tail(1)
# Calculate nbr of deaths per million people
df_deaths_per_million = pd.DataFrame(columns=df_deaths.columns, index=df_cases.index)
for x in df_deaths_per_million.columns:
  df_deaths_per_million[x] = 1000000 * df_deaths[x] // pop[x]

print('Nbr of covid-19 deaths per million people')
df_deaths_per_million.tail(1)
fig, ax = plt.subplots(1,2, figsize=(28,6))

# Axis 0: cases per million
for x in df_cases.columns:
  ax[0].bar(x, df_cases_per_million[x].tail(1))

# ax[0].set_ylabel('Number of cases per million')
ax[0].set_xticklabels(country_list, rotation=45, horizontalalignment='right')
ax[0].set_title('Covid-19 confirmed cases per million people as of ' + str(last_date), fontsize='x-large')

# Chart 2: deaths per million 
for x in df_cases.columns:
  ax[1].bar(x, df_deaths_per_million[x].tail(1))

# ax[1].set_ylabel('Number of deaths per million')
ax[1].set_xticklabels(country_list, rotation=45, horizontalalignment='right', fontsize='large')
ax[1].set_title('Covid-19 deaths per million people as of ' +  str(last_date), fontsize='x-large')

plt.show()
#########################################################################
# Prediction model parameters (Confirmed cases and deaths)
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases)
NBR_TRAIN_SAMPLES = NBR_SAMPLES - NBR_PREDICTIONS
NBR_TEST_SAMPLES = NBR_SAMPLES - NBR_TRAIN_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 50

print('Prediction model parameters for confirmed cases and deaths')
print('..........................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_TEST_SAMPLES: ', NBR_TEST_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('NBR_FEATURES: ', NBR_FEATURES)
print('TIME_STEPS:', TIME_STEPS)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('..........................................................')
# Process of CONFIRMED CASES 

# Split dataset into test and train subsets 
df_train_1 = df_cases_per_million.iloc[0:NBR_TRAIN_SAMPLES, 0:NBR_FEATURES] 
df_test_1 = df_cases.iloc[NBR_TRAIN_SAMPLES:, 0:NBR_FEATURES]

# Normalize test and train data (range: 0 - 1)
sc1 = MinMaxScaler(feature_range = (0, 1))
sc1.fit(df_train_1)
sc_df_train_1 = sc1.transform(df_train_1)
# sc_df_test = sc.transform(df_test)

# Prepare training sequences
X_train_1 = []
y_train_1 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_1.append(sc_df_train_1[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_1.append(sc_df_train_1[i, 0:NBR_FEATURES])
   
X_train_1, y_train_1 = np.array(X_train_1), np.array(y_train_1)
X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], X_train_1.shape[1], NBR_FEATURES))
# Build the RNN, dropout helps prevent overfitting

# Initialize structure
RNN1 = Sequential()

# Build layers: 2 LSTM layers with dropout
RNN1.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train_1.shape[1], NBR_FEATURES)))
RNN1.add(Dropout(0.25))
RNN1.add(LSTM(units = 256))
RNN1.add(Dropout(0.25))
RNN1.add(Dense(units = NBR_FEATURES, activation='elu'))

RNN1.summary()
%%time
# Compile the RNN 
RNN1.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the RNN
history_RNN1 = RNN1.fit(X_train_1, y_train_1, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=0)
# Convert the training history to a dataframe
history_RNN1_df = pd.DataFrame(history_RNN1.history)

# use Pandas native plot method
history_RNN1_df['loss'].plot(figsize=(8,5), title='MSE for "Confirmed Cases" neural network training (evaluation model)', xlabel='EPOCH', color='brown')
# Use now the full dataframe to predict / evaluate the model
df_full_1 = df_cases_per_million.copy()

# Scale full dataset (use same scaler fitted with train data earlier)
df_full_1 = sc1.transform(df_full_1)

X_test_1 = []
for i in range(NBR_TRAIN_SAMPLES, NBR_SAMPLES):
    X_test_1.append(df_full_1[i-TIME_STEPS:i, 0:NBR_FEATURES])

X_test_1 = np.array(X_test_1)
X_test_1 = np.reshape(X_test_1, (X_test_1.shape[0], X_test_1.shape[1], NBR_FEATURES))

# Make predictions
predicted_values_1 = RNN1.predict(X_test_1)
predicted_values_1 = sc1.inverse_transform(predicted_values_1)

# Reverse per million scaling
i = 0
for x in country_list:
  df_test_1[x + '_Predicted'] = predicted_values_1[:,i] * pop[x] / 1000000
  df_train_1[x] = df_train_1[x] * pop[x] / 1000000
  i = i + 1
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax[i,j].plot(df_train_1.index, df_train_1[x], color='navy', linewidth=1.5, label='Train')
  ax[i,j].plot(df_test_1.index, df_test_1[x], color='grey', linewidth=1.5, alpha=0.5, label='Test')
  ax[i,j].plot(df_test_1.index, df_test_1[x + '_Predicted'], color='navy', linestyle=':', linewidth=2.5, label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day prediction of covid-19 cases vs. training and validation data', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Process of DEATHS 
# Split dataset into test and train subsets 
df_train_2 = df_deaths_per_million.iloc[0:NBR_TRAIN_SAMPLES, 0:NBR_FEATURES] 
df_test_2 = df_deaths.iloc[NBR_TRAIN_SAMPLES:, 0:NBR_FEATURES]

# Normalize test and train data (range: 0 - 1)
sc2 = MinMaxScaler(feature_range = (0, 1))
sc2.fit(df_train_2)
sc_df_train_2 = sc2.transform(df_train_2)

# Prepare training sequences
X_train_2 = []
y_train_2 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_2.append(sc_df_train_2[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_2.append(sc_df_train_2[i, 0:NBR_FEATURES])
   
X_train_2, y_train_2 = np.array(X_train_2), np.array(y_train_2)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], NBR_FEATURES))
# Build the RNN, dropout helps prevent overfitting

# Initialize structure
RNN2 = Sequential()

# Build layers: 3 LSTM layers with dropout
RNN2.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train_2.shape[1], NBR_FEATURES)))
RNN2.add(Dropout(0.25))
RNN2.add(LSTM(units = 256))
RNN2.add(Dropout(0.25))
RNN2.add(Dense(units = NBR_FEATURES, activation='elu'))

RNN2.summary()
%%time
# Compile the RNN
RNN2.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the RNN
history_RNN2 = RNN2.fit(X_train_2, y_train_2, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=0)
# Convert the training history to a dataframe
history_RNN2_df = pd.DataFrame(history_RNN2.history)

# use Pandas native plot method
history_RNN2_df['loss'].plot(figsize=(8,5), title='MSE for "Deaths" neural network training (evaluation model)', xlabel='EPOCH', color='brown')
# Use now the full dataframe to predict / evaluate the model
df_full_2 = df_deaths_per_million.copy()

# Scale full dataset (use same scaler fitted with train data earlier)
df_full_2 = sc2.transform(df_full_2)

X_test_2 = []
for i in range(NBR_TRAIN_SAMPLES, NBR_SAMPLES):
    X_test_2.append(df_full_2[i-TIME_STEPS:i, 0:NBR_FEATURES])

X_test_2 = np.array(X_test_2)
X_test_2 = np.reshape(X_test_2, (X_test_2.shape[0], X_test_2.shape[1], NBR_FEATURES))

# Make predictions
predicted_values_2 = RNN2.predict(X_test_2)
predicted_values_2 = sc2.inverse_transform(predicted_values_2)

# Reverse per million scaling
i = 0
for x in country_list:
  df_test_2[x + '_Predicted'] = predicted_values_2[:,i] * pop[x] / 1000000
  df_train_2[x] = df_train_2[x] * pop[x] / 1000000
  i = i + 1
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax[i,j].plot(df_train_2.index, df_train_2[x], color='coral', linewidth=1.5, label='Train')
  ax[i,j].plot(df_test_2.index, df_test_2[x], color='grey', linewidth=1.5, alpha=0.5, label='Test')
  ax[i,j].plot(df_test_2.index, df_test_2[x + '_Predicted'], color='coral', linestyle=':', linewidth=2.5, label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day prediction of covid-19 deaths vs. training and validation data', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()
#########################################################################
# Future prediction model parameters for confirmed cases and deaths
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases)
NBR_TRAIN_SAMPLES = NBR_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 50

print('Future prediction model parameters for confirmed cases and deaths')
print('.................................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('TIME_STEPS:', TIME_STEPS)
print('NBR_FEATURES: ', NBR_FEATURES)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('.................................................................')
# Use full dataset as train data - CONFIRMED CASES
df_train_1 = df_cases_per_million.copy()

# Create empty dataframe with NBR_PREDICTIONS samples
start_date = df_train_1.index[-1] + timedelta(days=1)
ind = pd.date_range(start_date, periods=NBR_PREDICTIONS, freq='D')
df_pred_1 = pd.DataFrame(index=ind, columns=df_train_1.columns)
df_pred_1.fillna(value=0, inplace=True)

# Normalize train data (range: 0 - 1)
sc1 = MinMaxScaler(feature_range = (0, 1))
sc1.fit(df_train_1)
sc_df_train_1 = sc1.transform(df_train_1)

# Prepare training sequences
X_train_1 = []
y_train_1 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_1.append(sc_df_train_1[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_1.append(sc_df_train_1[i, 0:NBR_FEATURES])

X_train_1, y_train_1 = np.array(X_train_1), np.array(y_train_1)
X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], X_train_1.shape[1], NBR_FEATURES))
%%time
# Will reuse RNN1 already defined and validated earlier
RNN1.summary()

# Retrain the RNN with all available data
history_RNN1 = RNN1.fit(X_train_1, y_train_1, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=0)
# Convert the training history to a dataframe
history_RNN1_df = pd.DataFrame(history_RNN1.history)

# use Pandas native plot method
history_RNN1_df['loss'].plot(figsize=(8,5), title='MSE for "Confirmed Cases" neural network training (future predictive model)', xlabel='EPOCH', color='brown')
# Make predictions 
LSTM_predictions_scaled_1 = list()
batch = sc_df_train_1[-TIME_STEPS:]
current_batch = batch.reshape((1, TIME_STEPS, NBR_FEATURES))

for i in range(len(df_pred_1)):   
    LSTM_pred_1 = RNN1.predict(current_batch)[0]
    LSTM_predictions_scaled_1.append(LSTM_pred_1) 
    current_batch = np.append(current_batch[:,1:,:],[[LSTM_pred_1]],axis=1)
    
# Reverse downscaling
LSTM_predictions_1 = sc1.inverse_transform(LSTM_predictions_scaled_1)
df_pred_1 = pd.DataFrame(data=LSTM_predictions_1, index=df_pred_1.index, columns=df_pred_1.columns)

# Reverse per million scaling
for x in country_list:
    df_pred_1[x] = df_pred_1[x] * pop[x] / 1000000    
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax[i,j].plot(df_cases.index, df_cases[x], color='navy', linewidth=2, label='Actual data')
  ax[i,j].plot(df_pred_1.index, df_pred_1[x], color='navy', linewidth=3, linestyle=':', label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day future prediction of covid-19 confirmed cases by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()
# Use full dataset as train data - DEATHS
df_train_2 = df_deaths_per_million.copy()

# Create empty dataframe with NBR_PREDICTIONS samples
start_date = df_train_2.index[-1] + timedelta(days=1)
ind = pd.date_range(start_date, periods=NBR_PREDICTIONS, freq='D')
df_pred_2 = pd.DataFrame(index=ind, columns=df_train_2.columns)
df_pred_2.fillna(value=0, inplace=True)

# Normalize train data (range: 0 - 1)
sc2 = MinMaxScaler(feature_range = (0, 1))
sc2.fit(df_train_2)
sc_df_train_2 = sc2.transform(df_train_2)

# Prepare training sequences
X_train_2 = []
y_train_2 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_2.append(sc_df_train_2[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_2.append(sc_df_train_2[i, 0:NBR_FEATURES])

X_train_2, y_train_2 = np.array(X_train_2), np.array(y_train_2)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], NBR_FEATURES))
%%time
# Will reuse RNN2 already defined and validated earlier
RNN2.summary()

# Retrain the RNN with all available data
history_RNN2 = RNN2.fit(X_train_2, y_train_2, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=0)
# Convert the training history to a dataframe
history_RNN2_df = pd.DataFrame(history_RNN2.history)

# use Pandas native plot method
history_RNN2_df['loss'].plot(figsize=(8,5), title='MSE for "Confirmed Cases" neural network training (future predictive model)', xlabel='EPOCH', color='brown')
# Make predictions 
LSTM_predictions_scaled_2 = list()
batch = sc_df_train_2[-TIME_STEPS:]
current_batch = batch.reshape((1, TIME_STEPS, NBR_FEATURES))

for i in range(len(df_pred_2)):   
    LSTM_pred_2 = RNN2.predict(current_batch)[0]
    LSTM_predictions_scaled_2.append(LSTM_pred_2) 
    current_batch = np.append(current_batch[:,1:,:],[[LSTM_pred_2]],axis=1)
    
# Reverse downscaling
LSTM_predictions_2 = sc2.inverse_transform(LSTM_predictions_scaled_2)
df_pred_2 = pd.DataFrame(data=LSTM_predictions_2, index=df_pred_2.index, columns=df_pred_2.columns)

# Reverse per million scaling
for x in country_list:
    df_pred_2[x] = df_pred_2[x] * pop[x] / 1000000
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax[i,j].plot(df_deaths.index, df_deaths[x], color='coral', linewidth=2, label='Actual data')
  ax[i,j].plot(df_pred_2.index, df_pred_2[x], color='coral', linewidth=3, linestyle=':', label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day future prediction of covid-19 deaths by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()
#########################################################################
# Prediction model parameters (Confirmed cases growth rate)
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases_diff)
NBR_TRAIN_SAMPLES = NBR_SAMPLES - NBR_PREDICTIONS
NBR_TEST_SAMPLES = NBR_SAMPLES - NBR_TRAIN_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 100

print('Prediction model parameters for confirmed cases growth rates')
print('............................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_TEST_SAMPLES: ', NBR_TEST_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('NBR_FEATURES: ', NBR_FEATURES)
print('TIME_STEPS:', TIME_STEPS)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('............................................................')
# Process of CONFIRMED CASES GROWTH RATE data

# Split dataset into test and train subsets 
df_train_3 = df_cases_diff.iloc[0:NBR_TRAIN_SAMPLES, 0:NBR_FEATURES] 
df_test_3 = df_cases_diff.iloc[NBR_TRAIN_SAMPLES:, 0:NBR_FEATURES]

# Normalize test and train data (range: 0 - 1)
sc3 = MinMaxScaler(feature_range = (0, 1))
sc3.fit(df_train_3)
sc_df_train_3 = sc3.transform(df_train_3)

# Prepare training sequences
X_train_3 = []
y_train_3 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_3.append(sc_df_train_3[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_3.append(sc_df_train_3[i, 0:NBR_FEATURES])
   
X_train_3, y_train_3 = np.array(X_train_3), np.array(y_train_3)
X_train_3 = np.reshape(X_train_3, (X_train_3.shape[0], X_train_3.shape[1], NBR_FEATURES))
# Build the RNN, dropout helps prevent overfitting

# Initialize structure
RNN3 = Sequential()

# Build layers: 2 LSTM layers with dropout
RNN3.add(LSTM(units = 512, return_sequences = True, input_shape = (X_train_3.shape[1], NBR_FEATURES)))
RNN3.add(Dropout(0.25))
RNN3.add(LSTM(units = 512))
RNN3.add(Dropout(0.25))
RNN3.add(Dense(units = NBR_FEATURES, activation='elu'))

RNN3.summary()
%%time
# Compile the RNN
RNN3.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Retrain the RNN with all available data
history_RNN3 = RNN3.fit(X_train_3, y_train_3, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=0)
# convert the training history to a dataframe
history_RNN3_df = pd.DataFrame(history_RNN3.history)

# use Pandas native plot method
history_RNN3_df['loss'].plot(figsize=(8,5), title='MSE for "Cases Growth Rate" neural network training (evaluation model)', xlabel='EPOCH', color='brown')
# Use now the full dataframe to predict / evaluate the model
df_full_3 = df_cases_diff.copy()

# Scale full dataset (use same scaler fitted with train data earlier)
df_full_3 = sc3.transform(df_full_3)

X_test_3 = []
for i in range(NBR_TRAIN_SAMPLES, NBR_SAMPLES):
    X_test_3.append(df_full_3[i-TIME_STEPS:i, 0:NBR_FEATURES])

X_test_3 = np.array(X_test_3)
X_test_3 = np.reshape(X_test_3, (X_test_3.shape[0], X_test_3.shape[1], NBR_FEATURES))

# Make predictions
predicted_values_3 = RNN3.predict(X_test_3)
predicted_values_3 = sc3.inverse_transform(predicted_values_3)

i = 0
for x in country_list:
  df_test_3[x + '_Predicted'] = predicted_values_3[:,i]
  i = i + 1
# Plot future predictions of the cases growth rate
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax[i,j].plot(df_train_3.index, df_train_3[x], color='indigo', linewidth=1.5, label='Train')
  ax[i,j].plot(df_test_3.index, df_test_3[x], color='grey', linewidth=1.5, alpha=0.3, label='Test')
  ax[i,j].plot(df_test_3.index, df_test_3[x + '_Predicted'], color='indigo', linestyle=':', linewidth=2.5, label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='medium')
  if j<2: 
    j = j + 1
  else:
    i = i + 1
    j = 0


fig.suptitle(str(NBR_PREDICTIONS) + '-day prediction of the covid-19 cases growth rate by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()
#########################################################################
# Future prediction model parameters for confirmed cases growth rate
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases_diff)
NBR_TRAIN_SAMPLES = NBR_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 100

print('Future prediction model parameters for confirmed cases growth rate')
print('..................................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('TIME_STEPS:', TIME_STEPS)
print('NBR_FEATURES: ', NBR_FEATURES)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('..................................................................')
# Use full dataset as train data - CONFIRMED CASES GROWTH RATE
df_train_3 = df_cases_diff.copy()

# Create empty dataframe with NBR_PREDICTIONS samples
start_date = df_train_3.index[-1] + timedelta(days=1)
ind = pd.date_range(start_date, periods=NBR_PREDICTIONS, freq='D')
df_pred_3 = pd.DataFrame(index=ind, columns=df_train_3.columns)
df_pred_3.fillna(value=0, inplace=True)

# Normalize train data (range: 0 - 1)
sc3 = MinMaxScaler(feature_range = (0, 1))
sc3.fit(df_train_3)
sc_df_train_3 = sc3.transform(df_train_3)

# Prepare training sequences
X_train_3 = []
y_train_3 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_3.append(sc_df_train_3[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_3.append(sc_df_train_3[i, 0:NBR_FEATURES])

X_train_3, y_train_3 = np.array(X_train_3), np.array(y_train_3)
X_train_3 = np.reshape(X_train_3, (X_train_3.shape[0], X_train_3.shape[1], NBR_FEATURES))
%%time
# Will reuse RNN3 already defined and validated earlier
RNN3.summary()

# Retrain the RNN with all available data
history_RNN3 = RNN3.fit(X_train_3, y_train_3, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose=0)
# Convert the training history to a dataframe
history_RNN3_df = pd.DataFrame(history_RNN3.history)

# use Pandas native plot method
history_RNN3_df['loss'].plot(figsize=(8,5), title='MSE for "Cases Growth Rate" neural network training (future predictive model)', xlabel='EPOCH', color='brown')
# Make predictions 
LSTM_predictions_scaled_3 = list()
batch = sc_df_train_3[-TIME_STEPS:]
current_batch = batch.reshape((1, TIME_STEPS, NBR_FEATURES))

for i in range(len(df_pred_3)):   
    LSTM_pred_3 = RNN3.predict(current_batch)[0]
    LSTM_predictions_scaled_3.append(LSTM_pred_3) 
    current_batch = np.append(current_batch[:,1:,:],[[LSTM_pred_3]],axis=1)
    
# Reverse downscaling
LSTM_predictions_3 = sc3.inverse_transform(LSTM_predictions_scaled_3)
df_pred_3 = pd.DataFrame(data=LSTM_predictions_3, index=df_pred_3.index, columns=df_pred_3.columns)
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(21))
  ax[i,j].plot(df_train_3.index, df_train_3[x], color='indigo', linewidth=1.5, label='Actual data')
  ax[i,j].plot(df_pred_3.index, df_pred_3[x], color='indigo', linewidth=2.5, linestyle=':', label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='medium')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0


fig.suptitle(str(NBR_PREDICTIONS) + '-day future prediction of the confirmed cases growth rate by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()