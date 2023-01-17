import numpy as np 

import pandas as pd

import os

from pathlib import Path

import collections

import matplotlib.pyplot as plt
input_path = Path('../input/nfl-big-data-bowl-2021')
plays_df = pd.read_csv(input_path/'plays.csv')

players_df = pd.read_csv(input_path/'players.csv')

games_df = pd.read_csv(input_path/'games.csv')

#just pull 1 week for now

week1_df = pd.read_csv(input_path/'week1.csv')

plays_df.shape, players_df.shape, games_df.shape, week1_df.shape
plays_df.head()
#89 pass plays in this game!

one_game = plays_df[plays_df['gameId'] == 2018090600]

one_game.shape
one_game.iloc[0]
#the falcons had 7 more pass plays than the eagles this game

one_game['possessionTeam'].value_counts()
atl_plays = one_game[one_game['possessionTeam'] == 'ATL']

phi_plays = one_game[one_game['possessionTeam'] == 'PHI']
atl_plays.offenseFormation.value_counts(), phi_plays.offenseFormation.value_counts()
plt.hist([atl_plays.yardsToGo, phi_plays.yardsToGo],

        label=['Atl', 'Phi'])

plt.title('How many yards needed for a first on pass attempts')

plt.legend(loc='upper right');
plt.hist([atl_plays.defendersInTheBox, phi_plays.defendersInTheBox],

        label=['Atl', 'Phi'])

plt.title('How many defenders in the box is the passing team facing')

plt.legend(loc='upper right');
plt.hist([atl_plays.numberOfPassRushers, phi_plays.numberOfPassRushers],

        label=['Atl', 'Phi'])

plt.title('How many pass rushers is the passing team facing')

plt.legend(loc='upper right');
#on one play the eagles only had two defensive backs

phi_plays.personnelD.value_counts(), atl_plays.personnelD.value_counts()
#neither Foles or Ryan are mobile enough to be scrambling around often :)

atl_plays.typeDropback.value_counts(), phi_plays.typeDropback.value_counts()
plt.hist([atl_plays.absoluteYardlineNumber, phi_plays.absoluteYardlineNumber],

        label=['Atl', 'Phi'])

plt.title('How far down the field is each team on their pass plays')

plt.legend(loc='upper right');
plt.hist([atl_plays.playResult, phi_plays.playResult],

        label=['Atl', 'Phi'])

plt.title('Net yards gained per pass play')

plt.legend(loc='upper right');
plt.hist([atl_plays.epa, phi_plays.epa],

        label=['Atl', 'Phi'])

plt.title('Expected points added on the play -- avg of every next soring outcome')

plt.legend(loc='upper right');
players_df.head()
players_df['position'].value_counts()
#The big name college programs have the most players, as expected

players_df.collegeName.value_counts()
plt.hist(players_df.weight)

plt.title('weight of players involved in pass plays');
#now lets take a look at the game data

games_df.head()
#We can use the gameId and playID information to map this data onto our plays dataframe

week1_df.head()
sorted(week1_df.s, reverse=True)[0:5], sorted(week1_df.a, reverse=True)[0:5]
#lol the fastest speeds and accelerations recorderd are of the ball itself??

week1_df[week1_df.a >35].iloc[0:2]
week1_df_players = week1_df[week1_df.displayName != 'Football']

sorted(week1_df_players.s, reverse=True)[0:5], sorted(week1_df_players.a, reverse=True)[0:5]
week1_df_players[week1_df_players.s > 11]
week1_df_players[week1_df_players.a > 16]
atl_plays.iloc[0].gameId, atl_plays.iloc[0].playId
wk1_atl_phi = week1_df[week1_df['gameId'] == 2018090600]

wk1_atl_phi
one_play = wk1_atl_phi[wk1_atl_phi['playId'] == 75]

one_play
one_play['position'].value_counts()
#lets see the personnel on this same play

atl_plays.iloc[0].personnelO, atl_plays.illoc[0].personnelD