import numpy as np

import os 

import pandas as pd

import re

from dateutil.relativedelta import relativedelta



matches_re = re.compile('.*/atp_matches_.*\.csv')

players_re = re.compile('.*/atp_players\.csv')



matches_frames = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filepath = os.path.join(dirname, filename)

        if matches_re.match(filepath):

            matches_frames.append(pd.read_csv(filepath, delimiter = ','))

        elif players_re.match(filepath):

            players = pd.read_csv(filepath)

matches = pd.concat(matches_frames, ignore_index=True)

        

matches.dataframeName = 'Matches'

#print(matches.head())

#print(players.head())

matches = matches[['tourney_id', 'tourney_name', 'tourney_date', 'surface', 'winner_id', 'loser_id',  'score', 'best_of', 'round',

       'minutes']]

matches['tourney_date'] = pd.to_datetime(matches['tourney_date'], format='%Y%m%d')



winner_data = matches.join(players.set_index('player_id'), on='winner_id')

loser_data = matches.join(players.set_index('player_id'), on='loser_id')

winner_data['birthdate'] = pd.to_datetime(winner_data['birthdate'], format='%Y%m%d')

loser_data['birthdate'] = pd.to_datetime(loser_data['birthdate'], format='%Y%m%d')



matches[['w_hand', 'w_birthdate', 'w_country']] = winner_data[['hand', 'birthdate', 'country']]

matches['w_name'] = winner_data['name_first'] + ' ' + winner_data['name_list']

matches['l_name'] = loser_data['name_first'] + ' ' + loser_data['name_list']

matches[['l_hand', 'l_birthdate', 'l_country']] = loser_data[['hand', 'birthdate', 'country']]



def compute_age(tourney_date, birthdate):

    age = np.nan

    if 1 <= birthdate.month and birthdate.month <= 12:

        age = relativedelta(tourney_date, birthdate).years 

    return age

 

matches['w_age'] = matches.apply(lambda r: compute_age(r.tourney_date, r.w_birthdate), axis = 1)

matches['l_age'] = matches.apply(lambda r: compute_age(r.tourney_date, r.l_birthdate), axis = 1)



surface_map = {'Clay': 'Clay', 'Hard': 'Hard', 'Grass': 'Grass-Carpet', 'Carpet': 'Grass-Carpet'}

minutes = matches[['surface', 'minutes']].dropna()

minutes['surface'] = minutes['surface'].map(surface_map)

average_minutes = minutes.groupby('surface').mean()

r_l_matches = matches.query("w_hand != l_hand")

count_r_wins = r_l_matches[r_l_matches.w_hand == 'R'].shape[0]

count_l_wins = r_l_matches[(r_l_matches.w_hand == 'L') | (r_l_matches.w_hand == 'U')].shape[0]

perc_r_victories = 100 * count_r_wins / (count_r_wins + count_l_wins)

perc_l_victories = 100 * count_l_wins / (count_r_wins + count_l_wins)

oldies_wins = matches.query("w_age >= l_age + 20")[['tourney_date', 'w_age', 'w_name', 'l_age', 'l_name']]





print(average_minutes)

print('R Victories: {:.1f}%'.format(perc_r_victories))

print('L Victories: {:.1f}%'.format(perc_l_victories))

print(oldies_wins)