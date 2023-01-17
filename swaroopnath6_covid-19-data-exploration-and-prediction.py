import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
# Reading the data-set

dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

dataset['Date'] = dataset['Date'].apply(lambda date: pd.to_datetime(date, format = '%Y-%m-%d'))

dataset.head(5)
dataset['Province_State'] = dataset['Province_State'].fillna('NA')
inception_dates = dataset.where(dataset['ConfirmedCases'] > 0).groupby(['Country_Region', 'Province_State'])['Date'].min()

countries = list(inception_dates.index.get_level_values(0))

for country in countries:

    provinces = list(inception_dates[country].index.get_level_values(0))

    for province in provinces:

        df_index = dataset.loc[(dataset['Country_Region'] == country) & (dataset['Province_State'] == province)].index

        dataset.loc[df_index, 'date_of_inception'] = inception_dates[country, province]
dataset['days'] = dataset.apply(lambda row: (row['Date'] - row['date_of_inception']).days if row['date_of_inception'] != np.nan else np.nan, axis = 1)
data = dataset.loc[(dataset['Country_Region'] == 'Italy') & (dataset['days'] >= 0)]



# Italian govt imposed a lockdown on 9th of March

lockdown_date = pd.to_datetime('2020-03-09', format = '%Y-%m-%d')

day_of_lockdown = (lockdown_date - inception_dates.loc[('Italy', 'NA')]).days



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Confirmed Cases in Italy')

ax1.plot(data['days'], data['ConfirmedCases'])

ax1.set_ylabel('Confirmed Cases')

ax1.set_xlabel('Days since first case')

ax1.vlines(day_of_lockdown, ymin = 0, ymax = 150000, linestyle = '--', color = 'orange')

ax1.grid(True)



ax2.set_title('Fatalities in Italy')

ax2.plot(data['days'], data['Fatalities'])

ax2.set_ylabel('Fatalities')

ax2.set_xlabel('Days since first case')

ax2.vlines(day_of_lockdown, ymin = 0, ymax = 20000, linestyle = '--', color = 'orange')

ax2.grid(True)
daily_new_cases = dataset.loc[(dataset['Country_Region'] == 'Italy') & (dataset['days'] >= 0), 'ConfirmedCases'].diff()

daily_new_fatalities = dataset.loc[(dataset['Country_Region'] == 'Italy') & (dataset['days'] >= 0), 'Fatalities'].diff()

days = dataset.loc[(dataset['Country_Region'] == 'Italy') & (dataset['days'] >= 0), 'days']



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Daily Reported New Cases in Italy')

ax1.plot(days, daily_new_cases)

ax1.set_ylabel('New Cases')

ax1.set_xlabel('Days since first case')

ax1.vlines(day_of_lockdown, ymin = 0, ymax = 6000, linestyle = '--', color = 'orange')

ax1.grid(True)



ax2.set_title('Daily Reported Fatalities in Italy')

ax2.plot(days, daily_new_fatalities)

ax2.set_ylabel('New Fatalities')

ax2.set_xlabel('Days since first case')

ax2.vlines(day_of_lockdown, ymin = 0, ymax = 800, linestyle = '--', color = 'orange')

ax2.grid(True)
data = dataset.loc[(dataset['Country_Region'] == 'Italy') & (dataset['days'] >= 50)]



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Confirmed Cases in Italy')

ax1.plot(data['days'], data['ConfirmedCases'])

ax1.set_ylabel('Confirmed Cases')

ax1.set_xlabel('Days after 50th day from first reported case')

ax1.grid(True)



ax2.set_title('Fatalities in Italy')

ax2.plot(data['days'], data['Fatalities'])

ax2.set_ylabel('Fatalities')

ax2.set_xlabel('Days after 50th day from first reported case')

ax2.grid(True)
data = dataset.loc[(dataset['Country_Region'] == 'Italy') & (dataset['days'] >= 0)]



# Spain govt implicitly imposed a lockdown on 14th of March

lockdown_date = pd.to_datetime('2020-03-14', format = '%Y-%m-%d')

day_of_lockdown = (lockdown_date - inception_dates.loc[('Spain', 'NA')]).days



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Confirmed Cases in Spain')

ax1.plot(data['days'], data['ConfirmedCases'])

ax1.set_ylabel('Confirmed Cases')

ax1.set_xlabel('Days since first case')

ax1.vlines(day_of_lockdown, ymin = 0, ymax = 150000, linestyle = '--', color = 'orange')

ax1.grid(True)



ax2.set_title('Fatalities in Spain')

ax2.plot(data['days'], data['Fatalities'])

ax2.set_ylabel('Fatalities')

ax2.set_xlabel('Days since first case')

ax2.vlines(day_of_lockdown, ymin = 0, ymax = 20000, linestyle = '--', color = 'orange')

ax2.grid(True)
daily_new_cases = dataset.loc[(dataset['Country_Region'] == 'Spain') & (dataset['days'] >= 0), 'ConfirmedCases'].diff()

daily_new_fatalities = dataset.loc[(dataset['Country_Region'] == 'Spain') & (dataset['days'] >= 0), 'Fatalities'].diff()

days = dataset.loc[(dataset['Country_Region'] == 'Spain') & (dataset['days'] >= 0), 'days']



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Daily Reported New Cases in Spain')

ax1.plot(days, daily_new_cases)

ax1.set_ylabel('New Cases')

ax1.set_xlabel('Days since first case')

ax1.vlines(day_of_lockdown, ymin = 0, ymax = 6000, linestyle = '--', color = 'orange')

ax1.grid(True)



ax2.set_title('Daily Reported Fatalities in Spain')

ax2.plot(days, daily_new_fatalities)

ax2.set_ylabel('New Fatalities')

ax2.set_xlabel('Days since first case')

ax2.vlines(day_of_lockdown, ymin = 0, ymax = 800, linestyle = '--', color = 'orange')

ax2.grid(True)
data = dataset.loc[(dataset['Country_Region'] == 'Spain') & (dataset['days'] >= 50)]



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Confirmed Cases in Spain')

ax1.plot(data['days'], data['ConfirmedCases'])

ax1.set_ylabel('Confirmed Cases')

ax1.set_xlabel('Days after 50th day from first reported case')

ax1.grid(True)



ax2.set_title('Fatalities in Spain')

ax2.plot(data['days'], data['Fatalities'])

ax2.set_ylabel('Fatalities')

ax2.set_xlabel('Days after 50th day from first reported case')

ax2.grid(True)
us_provinces = dataset.loc[dataset['Country_Region'] == 'US', 'Province_State'].unique()



fig, axes = plt.subplots(18, 3, figsize = (30, 180))



axes = [ax for axis in axes for ax in axis]

for index, province in enumerate(us_provinces):

    data = dataset.loc[(dataset['Country_Region'] == 'US') & (dataset['Province_State'] == province)]

    axes[index].set_title('Confirmed Cases in US: {}'.format(province))

    axes[index].plot(data['days'], data['ConfirmedCases'])

    axes[index].set_ylabel('Confirmed Cases')

    axes[index].set_xlabel('Days since the first case')

    axes[index].grid(True)
fig, axes = plt.subplots(18, 3, figsize = (30, 180))



axes = [ax for axis in axes for ax in axis]

for index, province in enumerate(us_provinces):

    data = dataset.loc[(dataset['Country_Region'] == 'US') & (dataset['Province_State'] == province)]

    axes[index].set_title('Fatalities in US: {}'.format(province))

    axes[index].plot(data['days'], data['Fatalities'])

    axes[index].set_ylabel('Fatalities')

    axes[index].set_xlabel('Days since the first case')

    axes[index].grid(True)


fig, axes = plt.subplots(18, 3, figsize = (30, 150))



axes = [ax for axis in axes for ax in axis]

for index, province in enumerate(us_provinces):

    data = dataset.loc[(dataset['Country_Region'] == 'US') & (dataset['Province_State'] == province)]

    daily_new_cases = data['ConfirmedCases'].diff()

    axes[index].set_title('Daily Reported New Cases in US: {}'.format(province))

    axes[index].plot(data['days'], daily_new_cases)

    axes[index].set_ylabel('New Cases')

    axes[index].set_xlabel('Days since the first case')

    axes[index].grid(True)
fig, axes = plt.subplots(18, 3, figsize = (30, 150))



axes = [ax for axis in axes for ax in axis]

for index, province in enumerate(us_provinces):

    data = dataset.loc[(dataset['Country_Region'] == 'US') & (dataset['Province_State'] == province)]

    daily_new_cases = data['Fatalities'].diff()

    daily_new_fatalities = data['Fatalities'].diff()

    axes[index].set_title('Daily Reported New Fatalities in US: {}'.format(province))

    axes[index].plot(data['days'], daily_new_fatalities)

    axes[index].set_ylabel('New Cases')

    axes[index].set_xlabel('Days since the first case')

    axes[index].grid(True)
# Finding the states of US with confirmed case > 15000

us_provinces = dataset.loc[(dataset['Country_Region'] == 'US') & (dataset['ConfirmedCases'] > 15000), 'Province_State'].unique()



plt.figure(figsize = (15, 10))

italy_data = dataset[dataset['Country_Region'] == 'Italy']



for province in us_provinces:

    province_data = dataset[dataset['Province_State'] == province]

    plt.plot(province_data['days'], province_data['ConfirmedCases'], label = 'US: {}'.format(province))

    

plt.plot(italy_data['days'], italy_data['ConfirmedCases'], label = 'Italy')



plt.title('Comparison of Confirmed Cases between Italy and provinces of US')

plt.ylabel('Confirmed Cases')

plt.xlabel('Days since the first case')

    

plt.legend()
data = dataset.loc[(dataset['Country_Region'] == 'India') & (dataset['days'] >= 0)]



# Indian govt implicitly imposed a lockdown on 24th of March

lockdown_date = pd.to_datetime('2020-03-24', format = '%Y-%m-%d')

day_of_lockdown = (lockdown_date - inception_dates.loc[('India', 'NA')]).days



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Confirmed Cases in India')

ax1.plot(data['days'], data['ConfirmedCases'])

ax1.set_ylabel('Confirmed Cases')

ax1.set_xlabel('Days since first case')

ax1.vlines(day_of_lockdown, ymin = 0, ymax = 10000, linestyle = '--', color = 'orange')

ax1.grid(True)



ax2.set_title('Fatalities in India')

ax2.plot(data['days'], data['Fatalities'])

ax2.set_ylabel('Fatalities')

ax2.set_xlabel('Days since first case')

ax2.vlines(day_of_lockdown, ymin = 0, ymax = 300, linestyle = '--', color = 'orange')

ax2.grid(True)
daily_new_cases = dataset.loc[(dataset['Country_Region'] == 'India') & (dataset['days'] >= 0), 'ConfirmedCases'].diff()

daily_new_fatalities = dataset.loc[(dataset['Country_Region'] == 'India') & (dataset['days'] >= 0), 'Fatalities'].diff()

days = dataset.loc[(dataset['Country_Region'] == 'India') & (dataset['days'] >= 0), 'days']



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Daily Reported New Cases in India')

ax1.plot(days, daily_new_cases)

ax1.set_ylabel('New Cases')

ax1.set_xlabel('Days since first case')

ax1.vlines(day_of_lockdown, ymin = 0, ymax = 1000, linestyle = '--', color = 'orange')

ax1.grid(True)



ax2.set_title('Daily Reported Fatalities in India')

ax2.plot(days, daily_new_fatalities)

ax2.set_ylabel('New Fatalities')

ax2.set_xlabel('Days since first case')

ax2.vlines(day_of_lockdown, ymin = 0, ymax = 40, linestyle = '--', color = 'orange')

ax2.grid(True)
data = dataset.loc[(dataset['Country_Region'] == 'India') & (dataset['days'] >= 65)]



fix, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 8))

ax1.set_title('Confirmed Cases in India')

ax1.plot(data['days'], data['ConfirmedCases'])

ax1.set_ylabel('Confirmed Cases')

ax1.set_xlabel('Days after 65th day from first reported case')

ax1.grid(True)



ax2.set_title('Fatalities in India')

ax2.plot(data['days'], data['Fatalities'])

ax2.set_ylabel('Fatalities')

ax2.set_xlabel('Days after 65th day from first reported case')

ax2.grid(True)
italy_data = dataset[dataset['Country_Region'] == 'Italy']

india_data = dataset[dataset['Country_Region'] == 'India']



_, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 10))

ax1.plot(italy_data['days'], italy_data['ConfirmedCases'], label = 'Italy')

ax1.plot(india_data['days'], india_data['ConfirmedCases'], label = 'India')

ax1.set_title('Comparison of Confirmed Cases between India & Italy')

ax1.set_xlabel('Days since first case')

ax1.set_ylabel('Confirmed Cases')

ax1.set_ylim([0, 10000])

ax1.legend()





ax2.plot(italy_data['days'], italy_data['Fatalities'], label = 'Italy')

ax2.plot(india_data['days'], india_data['Fatalities'], label = 'India')

ax2.set_title('Comparison of Confirmed Cases between India & Italy')

ax2.set_xlabel('Days since first case')

ax2.set_ylabel('Confirmed Cases')

ax2.set_ylim([0, 500])

ax2.legend()