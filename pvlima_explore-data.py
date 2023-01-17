%matplotlib inline

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
races = pd.read_csv('../input/all_races.csv', nrows=146928)
races.head(2)
def clean_data(df):

    # fix one missing net time

    flt = (df.place==1)&(df.event=='family-race')&(df.event_year==2015)

    df.loc[flt, 'net_time'] = df[flt].official_time

    df.official_time = pd.to_timedelta(df.official_time)

    # many cases of missing net time

    df.net_time = np.where(df.net_time=='-', np.nan, df.net_time)

    df.net_time = pd.to_timedelta(df.net_time)

    # extract the date

    df['birth_year'] = df['birth_date'].str[6:].astype(int)

    return df

    

def input_missing_event_net_time(df):

    # if the runner is on top10, set missing net time equal to the official time

    df.net_time = np.where((df.net_time.isnull())&(df.place <= 10), 

                            df.official_time, df.net_time)

    df['delay_time'] = df['official_time'].dt.seconds - df['net_time'].dt.seconds

    df['delay_time_mean'] = df.delay_time.rolling(window=10, min_periods=5).mean()

    df['net_time_mean_sec'] = df['official_time'].dt.seconds - df['delay_time_mean']

    df['net_time'] = np.where(df.net_time.isnull(),

                            pd.to_timedelta(df.net_time_mean_sec, unit='s'), 

                            df.net_time)

    df = df.drop(['net_time_mean_sec','delay_time_mean','delay_time'], axis=1)

    assert not (df.official_time < df.net_time).any() 

    assert not df.net_time.isnull().any() 

    return df



def add_features(df):

    df['pace'] = df.net_time / df.distance

    return df
races = clean_data(races)

races = races.groupby(['event','event_year']).apply(input_missing_event_net_time)

races = add_features(races)
assert not races.net_time.isnull().any()
races[['distance','event','event_year']].drop_duplicates().groupby('distance').count()[['event']].add_suffix('_count')
races.shape[0], races.groupby(['name','birth_date']).count().shape[0]
g = races.groupby(['event','event_year']).count().reset_index().rename(columns={'place':'count'})

fig, ax = plt.subplots(4,2, figsize=(14,15))

ax = ax.ravel()

for i, race in enumerate(g.event.unique()):

    g[g.event==race].plot.bar(x='event_year', y='count', ax=ax[i], title=race)
def plot_avg_times_for_top(top_n):

    df = races[(races.place<=top_n)&(races.event!='family-race')].copy()

    # average time in minutes

    df.official_time = df.official_time.dt.seconds / 60.0

    df = df.groupby(['event','event_year'])[['official_time']].mean().reset_index()



    fig, ax = plt.subplots(3,2, figsize=(14,15), sharex=True)

    ax = ax.ravel()

    for i, ev in enumerate(df.event.unique()):

        df[df.event==ev].plot.scatter(x='event_year',y='official_time', ax=ax[i])

        ax[i].set_title(ev)



plot_avg_times_for_top(25)
plot_avg_times_for_top(500)
fig, ax = plt.subplots(2,2, figsize=(14,5*2))

ax = ax.ravel()

for i,ev in enumerate(['dia-do-pai','sao-joao','meia_maratona','maratona']):

    df = races[(races.event_year==2016)&(races.event==ev)].copy()

    sns.distplot(df.net_time.dt.seconds/60, hist=False, color="g", kde_kws={"shade": True}, ax=ax[i])

    sns.distplot(df.official_time.dt.seconds/60, hist=False, color="b", kde_kws={"shade": True}, ax=ax[i])

    ax[i].set_title(ev);
races[(races.event_year==2016)&(races.event=='dia-do-pai')].sex.value_counts()
fig, ax = plt.subplots(2,2, figsize=(14,5*2))

ax = ax.ravel()

for i,ev in enumerate(['dia-do-pai','sao-joao','meia_maratona','maratona']):

    df = races[(races.event_year==2016)&(races.event==ev)].copy()

    sns.distplot(df[df.sex=='M'].net_time.dt.seconds/60, hist=False, color="b", kde_kws={"shade": True}, ax=ax[i])

    sns.distplot(df[df.sex=='F'].net_time.dt.seconds/60, hist=False, color="r", kde_kws={"shade": True}, ax=ax[i])

    ax[i].set_title(ev);
before = races[(races.event_year<=2014)&(races.event=='maratona')]

after = races[(races.event_year>2014)&(races.event=='maratona')]

print('Average time from 2012-2014:', before.net_time.mean())

print('Average time from 2015-2016:', after.net_time.mean())
fig, ax = plt.subplots(1,1, figsize=(14,5))

sns.distplot(before.net_time.dt.seconds/60, hist=False, color="b", kde_kws={"shade": True}, ax=ax)

sns.distplot(after.net_time.dt.seconds/60, hist=False, color="g", kde_kws={"shade": True}, ax=ax);
RUNNER = 'Pedro Lima'

RUNNER_YEAR = 1974
def show_race_histogram(races, event, runner, runner_year):

    mm = races[(races.event==event)].event_year.describe()[['min','max']].astype(int)

    rng = list(range(mm['min'],mm['max']+1))

    nrows = math.ceil(len(rng)/2)

    fig, ax = plt.subplots(nrows,2, figsize=(14,5*nrows), sharex=True)

    ax = ax.ravel()

    for i, year in enumerate(rng):

        df = races[(races.event_year==year)&(races.event==event)].copy()

        df = df.sort_values(by='net_time').reset_index()

        df.net_time = df.net_time.dt.seconds/60

        df.net_time.hist(bins=60, ax=ax[i])

        runner_time = df[(df.name==runner)&(df.birth_year==runner_year)]

        if len(runner_time) > 0:

            ymin, ymax = ax[i].get_ylim()

            xpos = runner_time.net_time.values[0]

            txt = '{}%'.format(int(runner_time.index.values[0]/len(df)*100))

            ax[i].text(1.01*xpos, 0.97*ymax, txt)

            ax[i].vlines(xpos, ymin, ymax, colors='r')

        ax[i].set_title(str(year))
show_race_histogram(races, event='dia-do-pai', runner=RUNNER, runner_year=RUNNER_YEAR)
show_race_histogram(races, event='mar', runner=RUNNER, runner_year=RUNNER_YEAR)
show_race_histogram(races, event='meia_maratona', runner=RUNNER, runner_year=RUNNER_YEAR)