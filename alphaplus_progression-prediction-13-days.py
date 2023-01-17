# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



dataset=dataset.drop(['Province/State'], axis=1)

dataset=dataset.drop(['Last Update'], axis=1)

dataset=dataset.drop(['Deaths'], axis=1)

dataset=dataset.drop(['Recovered'], axis=1)

dataset.set_index('SNo',inplace = True)

dataset['ObservationDate'] = pd.to_datetime(dataset['ObservationDate']).astype(str)
China_Data = dataset['Country/Region'] == ('Mainland China')

covid_china_data = dataset[China_Data]

US_Data = dataset['Country/Region'] == ('US')

covid_US_data = dataset[US_Data]



India_Data = dataset['Country/Region'] == ('India')

covid_India_data = dataset[India_Data]

frames=[covid_china_data,covid_US_data,covid_India_data]

result = pd.concat(frames)

result.to_csv("result.csv")



frames_usInd=[covid_US_data,covid_India_data]

result_us_india=pd.concat(frames_usInd)



frames_ChinaIndia = [covid_china_data,covid_India_data]

result_china_india = pd.concat(frames_ChinaIndia)



frames_ChinaUS = [covid_china_data,covid_US_data]

result_china_US = pd.concat(frames_ChinaUS)

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

fig,ax = plt.subplots( figsize=(15,7), sharey=True)

result.groupby(['ObservationDate','Country/Region']).sum()['Confirmed'].unstack().plot(ax=ax)

ax.xaxis.set_minor_locator(AutoMinorLocator(4))



fig,ax = plt.subplots( figsize=(15,7), sharey=True)

result_us_india.groupby(['ObservationDate','Country/Region']).sum()['Confirmed'].unstack().plot(ax=ax)

ax.set_ylim(0, 200)

ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.xaxis.set_minor_locator(AutoMinorLocator(2))





fig,ax = plt.subplots( figsize=(15,7), sharey=True)

result_china_india.groupby(['ObservationDate','Country/Region']).sum()['Confirmed'].unstack().plot(ax=ax)



ax.yaxis.set_minor_locator(MultipleLocator(1000))

ax.xaxis.set_minor_locator(AutoMinorLocator(2))





fig,ax = plt.subplots( figsize=(15,7), sharey=True)

result_china_US.groupby(['ObservationDate','Country/Region']).sum()['Confirmed'].unstack().plot(ax=ax)



ax.yaxis.set_minor_locator(MultipleLocator(1000))

ax.xaxis.set_minor_locator(AutoMinorLocator(2))





confirm_data=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

death_data=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

columns=confirm_data.keys()



confirmed = confirm_data.loc[:, columns[27]:columns[-1]]             #Taking data from 15 Feb'2020  both for confirmed and death cases

deaths = death_data.loc[:, columns[27]:columns[-1]]



confirmed_country=confirm_data.loc[:, columns[1]].unique().astype(str)      #Taking all the unique countries

confirmed_country=confirmed_country[:49]

observe=confirmed.keys()

total_cases = []

total_country=[]

total_deaths = []

growth_rate = []                                                            

total_confirm=[]



for i in  observe:

    sum_confirm=confirmed[i].sum()

    sum_death=deaths[i].sum()

    total_cases.append(sum_confirm)

    total_deaths.append(sum_death)

    growth_rate.append((sum_death/sum_confirm)*100)

days_since_2_15 = np.array([i for i in range(len(observe))]).reshape(-1, 1)

total_cases = np.array(total_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_country=np.array(confirmed_country).reshape(-1,1)



days_in_future = 13

future_forcast = np.array([i for i in range(len(observe)+days_in_future)]).reshape(-1, 1)







start = '2/15/2020'

start_date = dt.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + dt.timedelta(days=i)).strftime('%m/%d/%Y'))





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(days_since_2_15, total_cases, test_size =0.20, random_state = 0)



from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10 , random_state=0)

lm=regressor.fit(X_train,y_train)



y_pred = regressor.predict(X_test)

predicted=regressor.predict(future_forcast)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.title('Comparison between Actual and predicted value')

plt.show()
ax=plt.axes()

plt.plot(y_test)

plt.plot(y_pred)

plt.title('Comparison between Actual and predicted value')

plt.show()

from matplotlib.pyplot import figure

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes()

plt.plot(confirmed_country,predicted)

ax.xaxis.set_major_locator(MultipleLocator(5))

plt.xlabel('First  46 countries from dataset')

plt.ylabel('Predicted values for next 13 days')

plt.title('Progression in next 13 days')

plt.show()



print('Future forcast dates:\n',future_forcast_dates)

figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes()

confirmedfirst30_country = confirmed_country[:36]

plt.plot(confirmedfirst30_country,growth_rate)

ax.xaxis.set_major_locator(MultipleLocator(5))

plt.xlabel('First  36 countries from dataset')

plt.ylabel('Growth Rate')

plt.title('Current country wise progression')

plt.show()

print('Model Score',lm.score(X_test, y_test))





# Predicting Model Accuracy

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)

print('Model Accuracy',accuracies.mean())

from sklearn.metrics import mean_squared_error, mean_absolute_error

print('MSE:',mean_squared_error(y_pred, y_test))

print('MAE:', mean_absolute_error(y_pred, y_test))
