import pandas as pd

import os

import numpy as np

from IPython.display import display, HTML

import multiprocessing as mp

import gc
def load_dataframe(year):

    dir = f'../input/playbyplay_{year}'

    players_df = pd.read_csv(f'{dir}/Players_{year}.csv', encoding = "ISO-8859-1")

    events_df = pd.read_csv(f'{dir}/Events_{year}.csv', encoding = "ISO-8859-1")

    return players_df, events_df
years = np.arange(2010, 2019)



with mp.Pool(4) as pool: 

    dfs = pool.map(load_dataframe, years)



dfs = list(zip(*dfs))

players = pd.concat(dfs[0])

events = pd.concat(dfs[1])



del dfs

gc.collect()



display(HTML(f'<h3>Players</h3>'))

display(players.sample(5))

display(players.describe(include="all").T)

display(HTML(f'<h3>Events</h3>'))

display(events.sample(5))

display(events.describe(include="all").T)
def minutes_played(group, disp=False):

    group = group.sort_values('ElapsedSeconds')

    last_event = group.tail(1)['EventType'].values[0]

    if last_event == 'sub_in':

        group.loc[0, ['ElapsedSeconds', 'EventType']] = (48*60, 'sub_out')

    group['Duration'] = group['ElapsedSeconds'].diff(1).fillna(group['ElapsedSeconds'])

    if disp:

        display(group)

    duration = group.loc[group['EventType'] == 'sub_out', 'Duration'].sum()

    return duration / 60
groups = events.loc[events['EventType'].isin(['sub_in', 'sub_out'])].groupby(['Season', 'DayNum', 'EventTeamID', 'EventPlayerID'])

with mp.Pool(4) as pool:

    min_played = pool.map(minutes_played, [group for _, group in groups])
mins_played = groups['EventID'].count().to_frame().reset_index()

mins_played['MinutesPlayed'] = min_played

display(mins_played.head(5))
ev = events.loc[(events['EventType'].isin(['sub_in', 'sub_out'])) & (events['EventPlayerID']==602324)]

%prun minutes_played(ev, disp=True)
events.loc[(events['EventPlayerID']==602324)].sort_values('ElapsedSeconds')