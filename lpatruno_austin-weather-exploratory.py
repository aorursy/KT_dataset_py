from datetime import datetime



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



pd.options.display.max_columns = 100



%matplotlib inline
df = pd.read_csv('austin_weather.csv')

df.replace('-', np.nan, inplace=True)

df.replace('T', np.nan, inplace=True)



for col in df:

    if col != 'Date' and col != 'Events':

        df[[col]] = df[[col]].astype(float)

        

df['Date'] = pd.to_datetime(df['Date'])

df.index = df['Date']

del df['Date']
# Fancy pandas datetimeindexing

df[datetime(2014, 5, 3):datetime(2014, 5, 6)]
# I love it :D

df['5/3/2014':'5/6/2014']
df[['TempHighF', 'TempAvgF', 'TempLowF']].resample('W').mean().plot(title='Weekly Temperature', figsize=(16, 6))



plt.ylabel('Temp (F)')

plt.show()
df[['PrecipitationSumInches']].resample('W').mean().plot(title='Weekly Precipitation', figsize=(16, 6))



plt.ylabel('Precipitation (inches)')

plt.show()
df[['HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent']].resample('M').mean().plot(title='Monthly Humidity', figsize=(16, 6))



plt.ylabel('Humidity')

plt.show()
df[['HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent']].plot.box()

plt.show()
q3 = df[['HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent']].quantile(.75)

q1 = df[['HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent']].quantile(.25)



iqr = q3-q1



iqr
df[['WindHighMPH', 'WindAvgMPH']].resample('W').mean().plot(title='Weekly Wind', figsize=(16, 6))



plt.ylabel('Wind (MPH)')

plt.show()
plt.scatter(df['WindAvgMPH'], df['PrecipitationSumInches'], alpha=.1)



plt.title('WindAvgMPH vs PrecipitationSumInches')

plt.xlabel('WindAvgMPH')

plt.ylabel('PrecipitationSumInches')

plt.show()
mask = df['WindAvgMPH'] < df['WindAvgMPH'].quantile(.5)



lower = df[mask]

higher = df[~mask]



plt.title('Distributions of daily precipitation')



lower['PrecipitationSumInches'].plot.hist(bins=20, histtype='step', label='Wind < 5 mph')

higher['PrecipitationSumInches'].plot.hist(bins=20, histtype='step', label='Wind >= 5 mph')



plt.legend()

plt.yscale('log')

plt.show()
# There has to be a better way of doing this

# I don't want subplots, I want all of the plots in one chart

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5), sharey=True)



df[(df['Rain'] == 1) | (df['Thunderstorm'] == 1)]['PrecipitationSumInches'].plot.box(ax=axes[0])

axes[0].set_title('Rain or Thunderstorm')



df[df['No Event'] == 1]['PrecipitationSumInches'].plot.box(ax=axes[1])

axes[1].set_title('No Event')

    



plt.show()
plt.title('Histogram of daily average wind (mph)')

df[(df['Rain'] == 1) | (df['Thunderstorm'] == 1)]['WindAvgMPH'].plot.hist(

    normed=True, histtype='step', label='Rain and/or Thunder')



df[df['No Event'] == 1]['WindAvgMPH'].plot.hist(

    normed=True, histtype='step', label='No event')



plt.xlabel('Wind (mph)')

plt.legend()

plt.show()
events = ['Rain', 'Thunderstorm', 'Fog', 'Snow']



for event in events:

    df[event] = False

    

for event in events:

    mask = df['Events'].apply(lambda row: event in row)

    df.loc[mask, event] = True

    

df['No Event'] = False

df.loc[~df[events].any(axis=1), 'No Event'] = True



events.append('No Event')
event_freq = df[events].sum()

event_freq.sort_values(ascending=False, inplace=True)



event_freq_rel = event_freq / df.shape[0]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))



event_freq.plot.bar(ax=axes[0]); axes[0].set_title('Event Frequency');

event_freq_rel.plot.bar(ax=axes[1]); axes[1].set_title('Event Relative Frequency');
event_stats = pd.concat([event_freq, event_freq_rel], axis=1)

event_stats.columns = ['Count', 'Frequency']

event_stats
# There has to be a better way of doing this

# I don't want subplots, I want all of the plots in one chart

fig, axes = plt.subplots(nrows=1, ncols=len(events), figsize=(18, 5), sharey=True)



for i, event in enumerate(events):

    df[df[event] == True]['VisibilityAvgMiles'].plot.box(ax=axes[i])

    axes[i].set_title(event)
print('Mean TempAvgF', df['TempAvgF'].mean())

print('Stdev TempAvgF', df['TempAvgF'].std())



plt.title('Distribution of TempAvgF')

bins = plt.hist(df['TempAvgF'], 15)

plt.show()
print('Temperature | Counts')

list(zip(bins[1], bins[0]))
print('Mean TempHighF', df['TempHighF'].mean())

print('Stdev TempHighF', df['TempHighF'].std())



plt.title('Distribution of TempHighF')

plt.hist(df['TempHighF'], 25)

plt.show()
print('Mean TempLowF', df['TempLowF'].mean())

print('Stdev TempLowF', df['TempLowF'].std())



plt.title('Distribution of TempLowF')

plt.hist(df['TempLowF'], 30)

plt.show()
bins = 25



plt.figure(figsize=(8,6))



df['TempLowF'].plot.hist(bins, normed=True, histtype='step', label='TempLowF')

df['TempAvgF'].plot.hist(bins, normed=True, histtype='step', label='TempAvgF')

df['TempHighF'].plot.hist(bins, normed=True, histtype='step', label='TempHighF')



plt.title('Distribution of daily temperatures')

plt.xlabel('Temperature (F)')

plt.legend()

plt.show()
plt.title('TempHighF vs TempLowF')

plt.scatter(df['TempLowF'], df['TempHighF'], alpha=.1)

plt.xlabel('TempLowF')

plt.ylabel('TempHighF')

plt.show()
df['TempRange'] = df['TempHighF'] - df['TempLowF']



plt.title('Distribution of daily temperature ranges')

plt.hist(df['TempRange'], 15)

plt.show()
# remove nan rows

non_nan_HumidityAvgPercent = df[~np.isnan(df['HumidityAvgPercent'])]['HumidityAvgPercent']



plt.title('HumidityAvgPercent')

plt.hist(non_nan_HumidityAvgPercent, 50)

plt.show()
plt.title('Distribution of PrecipitationSumInches (log y)')

plt.hist(df[~df['PrecipitationSumInches'].isnull()]['PrecipitationSumInches'], 10)

plt.yscale('log')

plt.show()
plt.title('HumidityAvgPercent vs PrecipitationSumInches')



plt.scatter(df['PrecipitationSumInches'], df['HumidityAvgPercent'])

plt.xlabel('PrecipitationSumInches')

plt.ylabel('HumidityAvgPercent')

plt.show()
plt.title('VisibilityAvgMiles vs HumidityAvgPercent')



plt.scatter(df['VisibilityAvgMiles'], df['HumidityAvgPercent'])

plt.xlabel('VisibilityAvgMiles')

plt.ylabel('HumidityAvgPercent')

plt.show()
plt.title('VisibilityAvgMiles vs DewPointAvgF')



plt.scatter(df['VisibilityAvgMiles'], df['DewPointAvgF'])

plt.xlabel('VisibilityAvgMiles')

plt.ylabel('DewPointAvgF')

plt.show()
plt.title('VisibilityAvgMiles vs PrecipitationSumInches')



plt.scatter(df['VisibilityAvgMiles'], df['PrecipitationSumInches'])

plt.xlabel('VisibilityAvgMiles')

plt.ylabel('PrecipitationSumInches')

plt.show()