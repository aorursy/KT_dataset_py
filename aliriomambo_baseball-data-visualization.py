

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Loading Files to Dataframe

import os

game_frames = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for game_file in filenames:

        #print(os.path.join(dirname, filename))

        game_frame = pd.read_csv(os.path.join(dirname, game_file), names=['type','multi2','multi3','multi4','multi5','multi6','event'])

        game_frames.append(game_frame)
games = pd.concat(game_frames)





games.loc[games['multi5'] == '??', ['multi5']] = ''



identifiers = games['multi2'].str.extract(r'(.LS(\d{4})\d{5})')

identifiers = identifiers.fillna(method='ffill')

identifiers.columns = ['game_id', 'year']

games = pd.concat([games, identifiers], axis=1, sort=False)

games = games.fillna(' ')

games.loc[:, 'type'] = pd.Categorical(games.loc[:, 'type'])

# Data Transformation

attendance = games.loc[(games['type'] == 'info') & (games['multi2'] == 'attendance'),['year','multi3']]

attendance.columns = ['year','attendance']

attendance.loc[:,'attendance'] = pd.to_numeric(attendance.loc[:,'attendance'])

attendance.sort_values(by=['year'], inplace=True)



# Visualization

attendance.plot(x = 'year',y ='attendance', figsize = (20,7), kind = 'bar')

plt.xlabel('Year')

plt.ylabel('Attendance')

plt.axhline(y=attendance['attendance'].mean(),color = 'green', linestyle = '--', label = 'Mean')





# Data Transformation

plays = games[(games['type']=='play')]

strike_outs = plays[plays['event'].str.contains('K')]

strike_outs = strike_outs.groupby(['year', 'game_id']).size()

strike_outs = strike_outs.reset_index(name='strike_outs')

strike_outs = strike_outs.loc[:,['year','strike_outs']].apply(pd.to_numeric)



# Visualization

strike_outs.plot(x='year',y ='strike_outs',kind='scatter').legend(['Strike Outs'])

plt.show()
#Offense Data Transformation

plays = games[(games['type']=='play')]

plays.columns = ['type','inning','team','player','count','pitches','event','game_id','year']

hits = plays.loc[plays['event'].str.contains('^(?:S(?!B)|D|T|HR)'),['inning','event']]



hits.loc[:, 'inning'] = pd.to_numeric(hits.loc[:, 'inning'])



replacements = {r'^S(.*)': 'single',

r'^D(.*)': 'double',

r'^T(.*)': 'triple',

r'^HR(.*)': 'hr'}

hit_type = hits['event'].replace(replacements, regex=True)

hits = hits.assign(hit_type = hit_type)

hits = hits.groupby(['inning','hit_type']).size().reset_index(name = 'count')



hits['hit_type'] = pd.Categorical(hits['hit_type'],['single','double','triple','hr'])



hits = hits.sort_values(['inning','hit_type'])



hits = hits.pivot(index='inning', columns='hit_type', values ='count')



# Visualization

hits.plot.bar(stacked=True)

plt.show()

# Frames

plays_frame = games.query("type == 'play' & event != 'NP'")

plays_frame.columns = ['type', 'inning', 'team', 'player', 'count', 'pitches', 'event', 'game_id', 'year']



info = games.query("type == 'info' & (multi2 == 'visteam' | multi2 == 'hometeam')")

info = info.loc[:, ['year', 'game_id', 'multi2', 'multi3']]

info.columns = ['year', 'game_id', 'team', 'defense']

info.loc[info['team'] == 'visteam', 'team'] = '1'

info.loc[info['team'] == 'hometeam', 'team'] = '0'

info = info.sort_values(['year', 'game_id', 'team']).reset_index(drop=True)



my_mask = plays_frame['event'].str.contains('^\d+')

my_mask_b = plays_frame['event'].str.contains('E')

my_mask_c = plays_frame['event'].str.contains('^(?:P|C|F|I|O)')

events = plays_frame.query("~(@my_mask & ~@my_mask_b) & ~@my_mask_c ")



#events = events.query("~@my_mask_c")

events = events.drop(['type', 'player', 'count', 'pitches'], axis=1)

events = events.sort_values(['team', 'inning']).reset_index()

replacements = {

      r'^(?:S|D|T).*': 'H',

      r'^HR.*': 'HR',

      r'^W.*': 'BB',

      r'.*K.*': 'SO',

      r'^HP.*': 'HBP',

      r'.*E.*\..*B-.*': 'RO',

      r'.*E.*': 'E',

    }

event_type = events['event'].replace(replacements, regex=True)

events = events.assign(event_type=event_type)

events = events.groupby(['year', 'game_id', 'team', 'event_type']).size().reset_index(name='count')
# Data Transformation

plays = games.query("type == 'play' & event!= 'NP'")

plays.columns = ['type', 'inning', 'team', 'player', 'count', 'pitches', 'event', 'game_id', 'year']

pa = plays.loc[plays['player'].shift() != plays['player'],['year','game_id','inning','team','player']]

pa = pa.groupby(['year','game_id','team']).size().reset_index(name = 'PA')



events = events.set_index(['year','game_id','team','event_type'])

events = events.unstack().fillna(0).reset_index()



events.columns = events.columns.droplevel()

events.columns = ['year', 'game_id', 'team', 'BB', 'E', 'H', 'HBP', 'HR', 'ROE', 'SO']

events = events.rename_axis(None, axis = 'columns')

events_plus_pa = pd.merge(events, pa, how = 'outer', left_on = ['year','game_id','team'], right_on = ['year','game_id','team'])

defense = pd.merge(events_plus_pa,info)

defense.loc[:,'DER'] =  1 - ((defense['H'] + defense['ROE']) / (defense['PA'] - defense['BB'] - defense['SO'] - defense['HBP'] - defense['HR']))

defense.loc[:,'year'] = pd.to_numeric(defense['year'])



# Visualization

der = defense.loc[(defense['year'] >= 1978) , ['year','defense','DER']]

der = der.pivot(index = 'year', columns= 'defense', values = 'DER')



der.plot(x_compat = True, xticks = range(1978,2018,4), rot = 45)

plt.show()


