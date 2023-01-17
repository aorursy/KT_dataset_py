import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
dataset.shape
dataset.dtypes
dataset.isnull().sum()
dataset.head(10)
dataset['ObservationDate'] = pd.to_datetime(dataset['ObservationDate'])
dataset['Last Update'] = pd.to_datetime(dataset['Last Update'])
months = ['Jan','Feb','March','April(Till first week)']
countries_affected = []
for x in range(1,5):
    countries_affected.append(dataset[dataset['ObservationDate'].dt.month == x]['Country/Region'].nunique())
plt.figure(figsize = (15,5))    
plt.bar(months,countries_affected)    
plt.xlabel('Months')
plt.ylabel("Number of Countries")
plt.title('Number of Countries Getting Affected')
plt.show()
first_month = dataset[dataset['ObservationDate'] == '01/31/2020'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending = False).head(10)
second_month = dataset[dataset['ObservationDate'] == '02/29/2020'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending = False).head(10)
third_month = dataset[dataset['ObservationDate'] == '03/31/2020'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending = False).head(10)
fourth_month = dataset[dataset['ObservationDate'] == '04/07/2020'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending = False).head(10)
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15,8))
fig.tight_layout(pad=6.0)

ax[0,0].bar(first_month.index.tolist(),first_month.tolist())
ax[0,0].set_xticklabels(first_month.index.tolist(),rotation = 45, minor=False)
ax[0,0].title.set_text('January')

ax[0,1].bar(second_month.index.tolist(),second_month.tolist(), color = 'g')
ax[0,1].set_xticklabels(second_month.index.tolist(),rotation = 45, minor=False)
ax[0,1].title.set_text('February')

ax[1,0].bar(third_month.index.tolist(),third_month.tolist(), color = 'c')
ax[1,0].set_xticklabels(third_month.index.tolist(),rotation = 45, minor=False)
ax[1,0].title.set_text('March')

ax[1,1].bar(fourth_month.index.tolist(),fourth_month.tolist(), color = 'r')
ax[1,1].set_xticklabels(fourth_month.index.tolist(),rotation = 45, minor=False)
ax[1,1].title.set_text('April (Till first week)')

plt.show()

top_countries = fourth_month.index.tolist()
increase_rate = {}
for x in top_countries:
        increase_rate.update({x:((fourth_month_total[x] - third_month_total[x] )/ third_month_total[x]) * 100})
increase_rate = pd.DataFrame({'Countries':list(increase_rate.keys()), 'Increase Rate': list(increase_rate.values())})
increase_rate = increase_rate.sort_values(by = 'Increase Rate',ascending = False).head(10)
increase_rate
plt.figure(figsize = (15,8))
sns.barplot(x = 'Increase Rate', y = 'Countries', data = increase_rate)
plt.show()
third_month_total = dataset[dataset['ObservationDate'] == '03/31/2020'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending = False)
fourth_month_total = dataset[dataset['ObservationDate'] == '04/07/2020'].groupby(['Country/Region'])['Confirmed'].sum().sort_values(ascending = False)
countries_overall = dataset['Country/Region'].tolist()
increase_rate_overall = {}
for x in countries_overall:
    if (x not in third_month_total or x not in fourth_month_total):
        continue;
    else:
        increase_rate_overall.update({x:((fourth_month_total[x] - third_month_total[x] )/ third_month_total[x]) * 100})
increase_rate_overall = pd.DataFrame({'Countries':list(increase_rate_overall.keys()), 'Increase Rate': list(increase_rate_overall.values())})
increase_rate_overall = increase_rate_overall.sort_values(by = 'Increase Rate',ascending = False).head(10)
increase_rate_overall
plt.figure(figsize = (15,8))
sns.barplot(x = 'Increase Rate', y = 'Countries', data = increase_rate_overall)
plt.title('Top 10 Countries with the highest rate of increase in cases (Overall)')
plt.show()
april = dataset[dataset['ObservationDate'] == '04/07/2020']
us_cities_april = april[april['Country/Region'] == 'US'][['Province/State','Confirmed']].sort_values(by = 'Confirmed', ascending = False).head(10)
plt.figure(figsize = (15,8))
sns.barplot(x = 'Confirmed', y = 'Province/State', data = us_cities_april, color = 'cyan')
plt.title('Most Affected States in US')
plt.show()
us_dataset = dataset[dataset['Country/Region'] == 'US']
#Dropping these four days as cases were taken for Counties as opposed to the other days where cases were taken for States
us_dataset = us_dataset[(us_dataset['ObservationDate'] != '2020-03-06') & (us_dataset['ObservationDate'] != '2020-03-07') & (us_dataset['ObservationDate'] != '2020-03-08') & (us_dataset['ObservationDate'] != '2020-03-09')]
number_of_cities = us_dataset.groupby(['ObservationDate']).size()
plt.figure(figsize = (15,8))
plt.plot(number_of_cities, color = 'c', marker = 'o')
plt.title('Day wise number of cities getting affected (In US)')
plt.xlabel('Days')
plt.ylabel('Number of Cities')
plt.show()
fourth_month_recovery = dataset[dataset['ObservationDate'] == '04/07/2020'].groupby(['Country/Region'])['Recovered'].sum().sort_values(ascending = False)
for x in fourth_month_recovery.index.tolist():
    ratio = (fourth_month_recovery[x]/fourth_month_total[x]) * 100
    fourth_month_recovery[x] = ratio   
fourth_month_recovery = fourth_month_recovery.sort_values(ascending = False).head(10)    
plt.figure(figsize = (10,5))
plt.xticks(rotation = 45)
plt.xlabel('Countries')
plt.ylabel('Percentage of People recovered')
plt.title('Top Ten Countries with Highest Recovery rate')
plt.bar(fourth_month_recovery.index.tolist(), fourth_month_recovery.tolist(),color = 'g',)
plt.show()
fourth_month_deaths = dataset[dataset['ObservationDate'] == '04/07/2020'].groupby(['Country/Region'])['Deaths'].sum().sort_values(ascending = False)
for x in fourth_month_deaths.index.tolist():
    ratio = (fourth_month_deaths[x]/fourth_month_total[x]) * 100
    fourth_month_deaths[x] = ratio   
fourth_month_deaths = fourth_month_deaths.sort_values(ascending= False).head(10)
plt.figure(figsize = (10,5))
plt.xticks(rotation = 45)
plt.xlabel('Countries')
plt.ylabel('Percentage of Deaths')
plt.title('Top Ten Countries with Highest Death rate')
plt.bar(fourth_month_deaths.index.tolist(), fourth_month_deaths.tolist(),color = 'r',)
plt.show()
ind_confirmed = dataset[dataset['Country/Region'] == 'India'].groupby(['ObservationDate'])['Confirmed'].sum().tolist()
ind_recovered = dataset[dataset['Country/Region'] == 'India'].groupby(['ObservationDate'])['Recovered'].sum().tolist()
ind_deaths = dataset[dataset['Country/Region'] == 'India'].groupby(['ObservationDate'])['Deaths'].sum().tolist()
plt.figure(figsize = (15,8))
plt.plot(ind_confirmed, color = 'c', marker = 'o', label = 'Number of Cases')
plt.plot(ind_recovered, color = 'g', marker = 'o', label = 'Recovered')
plt.plot(ind_deaths, color = 'r', marker = 'o', label = 'Deaths')
plt.title('Covid-19 Cases in India')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.legend()
plt.show()
dataset_rnn = dataset.groupby(['ObservationDate']).agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})
dataset_rnn
dataset_rnn.shape
training_set = dataset_rnn.iloc[:,0:1].values

#Date Preprocessing
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating data structure with 45 timesteps 
X_train = []
y_train = []
for i in range(45, 60):
    X_train.append(training_set_scaled[i-45:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train) , np.array(y_train)   

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Initialize the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#Add first LSTM layer and Dropout regularisation
regressor.add(LSTM(units =50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding second layer
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding third layer
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding fourth layer
regressor.add(LSTM(units =50))
regressor.add(Dropout(0.2))

#Output layer
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mse')

#Training the model
#Taking a small batch size because the number of data points to train on is limited
regressor.fit(X_train, y_train, epochs = 50, batch_size = 5)
#Prediction and visualization
real_confirmed_cases = dataset_rnn.iloc[57:77,0:1].values

X_test = []

for i in range(57,77):
    X_test.append(training_set_scaled[i-45:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_confirmed_cases = regressor.predict(X_test)
predicted_confirmed_cases = sc.inverse_transform(predicted_confirmed_cases)
plt.figure(figsize = (12,8))
plt.plot(real_confirmed_cases, color='c',marker = 'o', label = 'Real Confirmed Cases')
plt.plot(predicted_confirmed_cases, color='g',marker = 'o', label = 'Predicted Number of Cases')
plt.title('Coronavirus Forecasting Trend in Cases (From 18th March to 7th April)')
plt.xlabel('Days')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()
