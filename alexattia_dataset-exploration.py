import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline
def myround(x, base=5):

    return int(base * round(float(x)/base))



df = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

df_global = pd.read_csv('../input/GlobalTemperatures.csv')

df['year'] = pd.to_datetime(df.dt).dt.year

df['5year'] = df['year'].apply(myround)

df_global['year'] = pd.to_datetime(df_global.dt).dt.year

df_global['5year'] = df_global['year'].apply(myround)

df['decade'] = df['year'].apply(lambda x:int(str(x)[:3]+'0'))

df_global['decade'] = df_global['year'].apply(lambda x:int(str(x)[:3]+'0'))
beginning_year = 1900

df_france = df[df['Country'] == 'France']



df_france_year = df_france.groupby('year').mean()

df_france_5year = df_france.groupby('5year').mean()

df_france_decade = df_france.groupby('decade').mean()



df_france_year[list(df_france_year.index).index(beginning_year):]['AverageTemperature'].plot(color='grey')

df_france_5year[list(df_france_5year.index).index(beginning_year):]['AverageTemperature'].plot(color='green')

df_france_decade[list(df_france_decade.index).index(beginning_year):]['AverageTemperature'].plot(color='red')
f, ax = plt.subplots(ncols=4, figsize=(20,6))

country = ['France', 'United States', 'South Korea']

for i, c in enumerate(country):

    df[df['Country'] == c].groupby('5year').mean()[8:]['AverageTemperature'].plot(ax=ax[i], label=c, 

                                                                                   color = (1,

                                                                                            np.random.randint(70, 200)/255.,

                                                                                            np.random.randint(10, 250)/255.))

    ax[i].set_title('Land Average Temperature in %s' % c)

    ax[i].set_ylabel('T in °C')

    ax[i].set_xlabel('')

df_global.groupby('5year').mean()['LandAverageTemperature'].plot(ax=ax[3], label='Global')

ax[3].set_title('Land Average Temperature on Earth')

ax[3].set_ylabel('T in °C')

ax[3].set_xlabel('')
def transform_ticks(tick):

    if len(tick) < 10:

        return tick

    elif len(tick.split(' ')) > 1:

        t = tick.split(' ')

        return ' '.join(t[:round(len(t)/2)]) + '\n' + ' '.join(t[round(len(t)/2):]) 

    else:

        return tick[:10]
period_time = 11

a = df.groupby(['Country', 'decade']).mean()

a['Diff%s' % period_time] = a.groupby(level=0).AverageTemperature.diff(periods=period_time)

d = a.groupby(level=0).last().sort_values('Diff%s' % period_time, ascending=False)[:15]['Diff%s' % period_time]
f, ax = plt.subplots(figsize=(20,6))

dd = sorted(d.to_dict().items(), key=lambda x:x[1], reverse=True)

countries, values = [x[0] for x in dd], [x[1] for x in dd]

sns.barplot(countries, values, ax = ax)

ax.set_ylabel('T °C difference between the 1900\'s and the 2010\'s')

ax.set_title('Countries where the climate change is the most important\nbetween the 1900\'s and the 2010\'s')

_ = ax.set_xticklabels([transform_ticks(b.get_text()) for b in ax.get_xticklabels()])