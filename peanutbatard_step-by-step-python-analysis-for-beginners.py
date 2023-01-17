import numpy as np 

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import os

import re
files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(str(os.path.join(dirname, filename)))

files
names = [re.findall('\w*.csv', x)[0].split('.')[0] for x in files]

names
data_dict = {}



for i, file in enumerate(files):

    name = names[i]

    data_dict[name] = pd.read_csv(file, encoding = 'latin-1') # https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte
data_dict['circuits'].sample(3)
data_dict['constructorResults'].sample(3)
data_dict['constructors'].sample(3)
data_dict['constructorStandings'].sample(3)
data_dict['drivers'].sample(3)
data_dict['driverStandings'].sample(3)
data_dict['lapTimes'].sample(3)
data_dict['pitStops'].sample(3)
data_dict['qualifying'].sample(3)
data_dict['races'].sample(3)
data_dict['races'][data_dict['races']['year'] == 2017]
data_dict['results'].sample(3)
data_dict['seasons'].sample(3)
data_dict['status'].sample(3)
data_dict['results'][data_dict['results']['raceId'] == 988].head()
data_dict['status'][data_dict['status']['statusId'].isin([1, 11, 36, 9])]
data_dict['results']['positionText'].value_counts()
data_dict['results'][data_dict['results']['positionText'].isin(['R', 'F', 'W', 'N', 'E'])].isna().sum()
data_dict['results'].drop(columns = ['rank', 'positionOrder'], inplace = True)



data_dict['constructors'].drop(columns = ['nationality', 'url', 'Unnamed: 5'], inplace = True)

data_dict['constructors'].rename(columns = {'name': 'constructorName'}, inplace = True)



data_dict['races'].drop(columns = ['time', 'url'], inplace = True)

data_dict['races'].rename(columns = {'name': 'raceName', 'date': 'raceDate'}, inplace = True)



data_dict['circuits'].drop(columns = ['lat', 'lng', 'alt', 'url'], inplace = True)

data_dict['circuits'].rename(columns = {'name': 'circuitName'}, inplace = True)



data_dict['drivers'].drop(columns = ['number', 'code', 'url'], inplace = True)
df = data_dict['results'].copy()

df = df.merge(data_dict['constructors'], how = 'left')

df = df.merge(data_dict['races'], how = 'left')

df = df.merge(data_dict['circuits'], how = 'left')

df = df.merge(data_dict['drivers'], how = 'left')

df = df.merge(data_dict['status'], how = 'left')
len(df), len(data_dict['results'])
df['year'].min(), df['year'].max()
df['fastestLapTimeSec'] = (df['fastestLapTime'].str.split(':', expand = True)[0].astype('float64') * 60) + df['fastestLapTime'].str.split(':', expand = True)[1].astype('float64')

df['fastestLapTimeSec'].tail(5)
df['fastestLapSpeed'].head(5)
df['fastestLapSpeed'].astype('float64')
df[df['fastestLapSpeed'] == '01:42.6'][['country', 'year', 'driverRef', 'fastestLapSpeed', 'fastestLapTime']]
df[(df['year'] == 2017) & (df['country'] == 'UAE')][['driverRef', 'fastestLapTime', 'fastestLapTimeSec', 'fastestLapSpeed']]
t = df.loc[23766, ['fastestLapTimeSec']] 

s = df.loc[23766, ['fastestLapSpeed']].astype('float64')

r = s[0] / t[0]

r
newSpeed = r * df.loc[23764, ['fastestLapTimeSec']][0]

df.loc[23764, ['fastestLapSpeed']] = str(newSpeed)

df[(df['year'] == 2017) & (df['country'] == 'UAE')][['driverRef', 'fastestLapTime', 'fastestLapTimeSec', 'fastestLapSpeed']]
df['fastestLapSpeed'] = df['fastestLapSpeed'].astype('float64')

df['fastestLapSpeedMetersPerSec'] = 1000 * df['fastestLapSpeed'] / 3600
df['fastestLapLength'] = df['fastestLapTimeSec'] * df['fastestLapSpeedMetersPerSec']

df['fastestLapLength'].tail(5)
df[(df['year'] == 2017) & (df['country'] == 'Italy')][['driverRef', 'circuitName', 'fastestLapLength']]
df[df['year'] == 2017].groupby(['driverRef']).agg({'points': 'sum'}).sort_values(by = 'points', ascending = False).plot.bar()
df[(df['year'] == 2017) & (df['surname'] == 'Ricciardo') & (df['country'] == 'Italy')][['fastestLapTime', 'fastestLapSpeed']]
df_speed = df.copy()
orig_df_speed_len = len(df_speed)
len(df_speed[df_speed['fastestLapSpeed'].isna()]) / len(df_speed)
df_speed[df_speed['fastestLapSpeed'].isna()].groupby(['year']).agg({'resultId': 'count'}).plot()
df_speed[df_speed['fastestLapSpeed'].isna()].groupby(['year']).agg({'resultId': 'count'}).loc[2000:]
df_speed = df_speed[df_speed['year'] >= 2004]
df_speed.isna().sum()
df_speed['fastestLapSpeed'].isna().sum() / len(df_speed)
df_speed[df_speed['fastestLapSpeed'].isna()].groupby(['year', 'country'], as_index = False).agg({'resultId': 'count'}).sort_values(by = 'resultId', ascending = False)
df_speed[(df_speed['country'] == 'China') & (df_speed['year'] == 2011)]
df_speed = df_speed[df_speed['raceId'] != 843]
df_speed[df_speed['fastestLapSpeed'].isna()]['status'].value_counts()
df_speed[(df_speed['fastestLapSpeed'].isna()) & (df_speed['status'] == 'Finished')]
df_speed.loc[22832]
df_kvy = data_dict['lapTimes'][(data_dict['lapTimes']['raceId'] == 941) & (data_dict['lapTimes']['driverId'] == 826)]

df_kvy.sort_values(by = 'milliseconds').head(5)
df_speed[(df_speed['country'] == 'Russia') & (df_speed['year'] == 2015)]['fastestLapLength'].median()
t = df_kvy.loc[106791, ['milliseconds']][0] / 1000

d = df_speed[(df_speed['country'] == 'Russia') & (df_speed['year'] == 2015)]['fastestLapLength'].median()

s = d / t

s = 3.6 * s

s
df_speed.loc[22832, ['fastestLapSpeed']] = s

df_speed[(df_speed['country'] == 'Russia') & (df_speed['year'] == 2015)][['driverRef', 'position', 'fastestLapSpeed']]
df_speed[df_speed['fastestLapSpeed'].isna()]['status'].value_counts()
df_speed['fastestLapSpeed'].isna().sum() / len(df_speed)
finisher = df_speed[(df_speed['status'].str.startswith('+')) | (df_speed['status'] == 'Finished')]['status'].drop_duplicates()

finisher = list(finisher)

finisher
len(df_speed[(df_speed['status'].isin(finisher))]) / len(df_speed)
df_speed = df_speed[(df_speed['status'].isin(finisher))]
df_speed.isna().sum()
1 - (len(df_speed) / orig_df_speed_len)
df_speed = df_speed.groupby(['year', 'country'], as_index = False).agg({'fastestLapSpeed': np.median, 'fastestLapLength': np.mean})

df_speed.rename(columns = {'fastestLapSpeed': 'medianFastestLapSpeed', 'fastestLapLength': 'approxLapLength'}, inplace = True)



countries = list(df_speed['country'].drop_duplicates())
fig, ax = plt.subplots(figsize = (5, 5))



country = 'Australia'



df_temp = df_speed[df_speed['country'] == country]



ax.scatter(df_temp['year'], df_temp['medianFastestLapSpeed'], color = '#4B878BFF')

ax.set(xlabel = 'year'

       , ylabel = 'median fasted lap speed (km/h)'

       , title = country

       , ylim = (0, 260)

       , xlim = (2003, 2018))



ax2 = ax.twinx()

ax2.plot(df_temp['year'], df_temp['approxLapLength'] / 1000, color = '#D01C1FFF')

ax2.set(xlabel = 'year'

        , ylabel = 'approx lap lenghth (km)'

        , ylim = (0, 6)

        , xlim = (2003, 2018))



ax2.spines['left'].set_color('#4B878BFF')

ax2.spines['right'].set_color('#D01C1FFF')



plt.tight_layout()
countries = list(df_speed['country'].drop_duplicates())

len(countries)
fig, ax = plt.subplots(nrows = 5, ncols = 5, figsize = (25, 20))



i = 0 

for r in range(0, 5):

    for c in range(0, 5):

        if i < len(countries):

            ax1 = ax[r, c]

            df_temp = df_speed[df_speed['country'] == countries[i]]

            # primary vertical axis

            ax1.scatter(df_temp['year'], df_temp['medianFastestLapSpeed'], color = '#4B878BFF')

            ax1.set(xlabel = 'year'

                    , ylabel = 'median fasted lap speed (km/h)'

                    , title = '{} GP: median fastest lap by season'.format(countries[i])

                    , ylim = (0, 260)

                    , xlim = (2003, 2018))

            # secondary vertical axis

            ax2 = ax1.twinx()

            ax2.plot(df_temp['year'], df_temp['approxLapLength'] / 1000, color = '#D01C1FFF')

            ax2.set(xlabel = 'year'

                    , ylabel = 'approx lap lenghth (km)'

                    , ylim = (0, 8)

                    , xlim = (2003, 2018))

            # colouring the axis

            ax2.spines['left'].set_color('#4B878BFF')

            ax2.spines['right'].set_color('#D01C1FFF')

            # colouring the ticks

            ax1.tick_params(axis='y', labelcolor = '#4B878BFF')

            ax2.tick_params(axis='y', labelcolor = '#D01C1FFF')

            # colouring the axis labels

            ax1.yaxis.label.set_color('#4B878BFF')

            ax2.yaxis.label.set_color('#D01C1FFF')

            # counter

            i = i + 1

        else:

            break

fig.tight_layout()

plt.show()
focus_countries = ['Australia', 'Brazil', 'Hungary', 'Malaysia', 'Monaco']

len(focus_countries)
for country in focus_countries:

    fig, ax = plt.subplots(figsize = (7, 7))

    df_temp = df_speed[df_speed['country'] == country]

    # scatter plot

    ax.scatter(df_temp['year'], df_temp['medianFastestLapSpeed'], color = '#4B878BFF')

    ax.set(xlabel = 'year'

            , ylabel = 'median fasted lap speed (km/h)'

            , title = '{} GP: speed by season (labels for 2004, 2014, 2017)'.format(country)

            , ylim = (0, 240)

            , xlim = (2002, 2018))

    # adding vertical line for 2014

    ax.axvline(x = 2014, color = '#a3a3a3', linewidth = 1, linestyle = 'dashed')

    # adding labels for the 3 key years

    plt.text(x = 2004

             , y = df_temp.iloc[0, 2] + 5

             , s = '{} km/h'.format(round(df_temp.iloc[0, 2], 1))

             , size = 8

             , color = '#4B878BFF'

             , ha = 'center')

    plt.text(x = 2014

             , y = df_temp.iloc[10, 2] - 15

             , s = '{} km/h'.format(round(df_temp.iloc[10, 2], 1))

             , size = 8

             , color = '#4B878BFF'

             , ha = 'center')

    plt.text(x = 2017

             , y = df_temp.iloc[13, 2] - 15

             , s = '{} km/h'.format(round(df_temp.iloc[13, 2], 1))

             , size = 9

             , color = '#4B878BFF'

             , ha = 'center')

plt.show()
# data stuff

df_fin = df.groupby(['nationality'], as_index = False).agg({'points': 'sum'}).sort_values('points', ascending = False).copy()

df_fin['points_pct'] = df_fin['points'] / df_fin['points'].sum()

df_fin = df_fin.iloc[0:10]



# viz stuff

import matplotlib.ticker as mtick

fig, ax = plt.subplots(figsize = (14, 5))

ax.bar(df_fin['nationality'], 100 * df_fin['points_pct'], color = '#4B878BFF')

ax.set(xlabel = 'nationality'

       , ylabel = '% of all points earned'

       , title = 'Top 10 nationalities in terms of percentage of championship points earned')

ax.get_children()[3].set_color('#D01C1FFF') 

plt.text(x = 3 - 0.15

         , y = 100 * df_fin['points_pct'].iloc[3] + 0.5

         , s = '{}%'.format(round(100 * df_fin['points_pct'].iloc[3], 2))

         , size = 10

         , color = '#D01C1FFF'

         , weight = 'bold')

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.show()
# data stuff

df_rel = df[['grid', 'position']].copy()

df_rel = df_rel.dropna()



# viz stuff

fig, ax = plt.subplots(figsize = (6, 6))

ax.scatter(df_rel['grid'], df_rel['position'], alpha = 0.05, color = '#4B878BFF')

ax.set(xlabel = 'grid position'

       , ylabel = 'final position'

       , title = 'Grid position vs final position')

plt.show()
# data stuff

df_cns = data_dict['constructorStandings'][['raceId', 'constructorId', 'points', 'position']].copy()

df_cns = df_cns.merge(data_dict['constructors'], how = 'left')

df_cns = df_cns.merge(data_dict['races'][['raceId', 'year', 'raceName', 'round']], how = 'left')

df_cns = df_cns[['constructorName', 'position', 'year', 'round']]

df_fnl = df_cns.groupby(['year'], as_index = False).agg({'round': np.max})

df_fnl.rename(columns = {'round': 'finalRound'}, inplace = True)

df_cns = df_cns.merge(df_fnl, how = 'left')

df_cns = df_cns[df_cns['round'] == df_cns['finalRound']]

df_cns = df_cns[['constructorName', 'position', 'year']]

focus_constructors = ['Mercedes', 'Ferrari', 'Red Bull', 'BRM', 'Tyrrell', 'Force India', 'Jordan']

focus_palette = {'Mercedes': '#707070'

                 , 'Ferrari': '#de0000'

                 , 'Red Bull': '#001496'

                 , 'BRM': '#228243'

                 , 'Tyrrell': '#4dc3eb'

                 , 'Force India': '#ff85ef'

                 , 'Jordan': '#fcd700'}

df_cns = df_cns[df_cns['constructorName'].isin(focus_constructors)]



# viz stuff

fig, ax = plt.subplots(figsize = (15, 5))

ax = sns.lineplot(data = df_cns

                  , x = 'year'

                  , y = 'position'

                  , hue = 'constructorName'

                  , palette = focus_palette

                  , hue_order = focus_constructors)

ax.set(xlabel = 'round'

       , ylabel = 'rank'

       , title = 'Final season ranking for few constructors since 1958')

ax.legend(loc = 'upper left')

from matplotlib.ticker import MaxNLocator

ax.xaxis.set_major_locator(MaxNLocator(integer = True))

ax.yaxis.set_major_locator(MaxNLocator(integer = True))

plt.show()