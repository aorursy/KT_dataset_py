import pandas as pd

from datetime import datetime



df = pd.read_csv('../input/ATP.csv')

df = df[df['tourney_date'] != 'tourney_date']

df['winner_id'] = pd.to_numeric(df['winner_id'])

df['loser_id'] = pd.to_numeric(df['loser_id'])
# Most majors

gs_winner_df = df[(df['tourney_level'] == 'G') & (df['round'] == 'F')]

grouped = gs_winner_df.groupby('winner_name').count().sort_values('winner_id', ascending= False).ix[:,0]

grouped
# List of all players in the data

winner_names = df[['winner_name','winner_id']].drop_duplicates().set_index('winner_id')

loser_names = df[['loser_name','loser_id']].drop_duplicates().rename(index=str, columns={"loser_name": "winner_name", "loser_id": "winner_id"}).set_index('winner_id')
# List of all players in the data

winner_names = df[['winner_name','winner_id']].drop_duplicates()

loser_names = df[['loser_name','loser_id']].drop_duplicates().rename(index=str, columns={"loser_name": "winner_name", "loser_id": "winner_id"})



player_unique_list = pd.concat([winner_names, loser_names])



player_unique_list['winner_id'] = pd.to_numeric(player_unique_list['winner_id'])



player_unique_list = player_unique_list.drop_duplicates(['winner_name','winner_id']).set_index('winner_id')



player_unique_list['elo_score'] = 1000
from dateutil.parser import parse



# Parse timestamp from tourney date

df['timestamp'] = df.loc[:,'tourney_date'].apply(lambda x: parse(str(x)))



# Setting timestamp as index

df = df.set_index(df['timestamp'])



columns_needed_for_elo = [

    'tourney_id',

    'tourney_name',

    'draw_size',

    'tourney_level',

    'winner_id',

    'winner_name',

    'loser_id',

    'loser_name',

    'score',

    'best_of',

    'round'

]



elo_df = df[columns_needed_for_elo]
def score_diff(score, best_of):

    if ((len(score.split()) == 3) and (best_of == 5)) or ((len(score.split()) == 2) and (best_of == 3)):

        return 1.75

    elif (len(score.split()) == 4) and (best_of == 5):

        return 1.5

    else:

        return 1
elo_df['score_diff'] = elo_df.apply(lambda x: score_diff(str(x['score']), x['best_of']), axis = 1)
def expected(A, B):

    """

    Calculate expected score of A in a match against B

    :param A: Elo rating for player A

    :param B: Elo rating for player B

    """

    return 1 / (1 + 10 ** ((B - A) / 400))





def elo(old, expected, score, score_diff, k=30):

    """

    Calculate the new Elo rating for a player

    :param old: The previous Elo rating

    :param exp: The expected score for this match

    :param score: The actual score for this match

    :param k: The k-factor for Elo (default: 32)

    :param score_diff: calculated score diff for result

    """

    return old + k * score_diff * (score - expected)
def elo_calc(row):

    prev_elo_winner = player_unique_list.loc[row['winner_id'],'elo_score']

    prev_elo_loser = player_unique_list.loc[row['loser_id'],'elo_score']

    

    exp_win = expected(prev_elo_winner, prev_elo_loser)

    exp_lose = expected(prev_elo_loser, prev_elo_winner)

    

    elo_winner = elo(prev_elo_winner, exp_win, 1, row['score_diff'])

    elo_loser = elo(prev_elo_loser, exp_lose, 0, row['score_diff'])

    

    player_unique_list.loc[row['winner_id'],'elo_score'] = elo_winner

    player_unique_list.loc[row['loser_id'],'elo_score'] = elo_loser

    

    #return [elo_winner, elo_loser]



    return pd.Series({'elo_winner': elo_winner, 'elo_loser':elo_loser}) 
elo_scores = elo_df.apply(lambda x: elo_calc(x), axis = 1)
elo_df['elo_loser'] = elo_scores.iloc[:,0]

elo_df['elo_winner'] = elo_scores.iloc[:,1]
grouped[grouped > 2].index.values.tolist()
gs_winner_list = grouped[grouped > 2].index.values.tolist()
%matplotlib inline

import matplotlib.pyplot as plt



fig = plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(1,1,1)

ax.set_title('ELO Plot without Weighing Tournament Types')

ax.set_xlabel('Year')

ax.set_ylabel('ELO Score')

ax.set_ylim([1000, 2200])



for player in gs_winner_list:

    player_inst = elo_df[elo_df['winner_name'] == player]['elo_winner']

    player_inst = player_inst.groupby(level=0).mean()

    player_inst.plot(label= player)



plt.grid(b=True, which='major', color='k', linestyle='--', linewidth = .2)

plt.legend(loc=9, bbox_to_anchor=(1.15, 1))
def elo_calc_with_k(row):

    prev_elo_winner = player_unique_list.loc[row['winner_id'],'elo_score']

    prev_elo_loser = player_unique_list.loc[row['loser_id'],'elo_score']

    

    exp_win = expected(prev_elo_winner, prev_elo_loser)

    exp_lose = expected(prev_elo_loser, prev_elo_winner)

    

    if row['tourney_level'] == 'G':

        k = 50

    elif row['tourney_level'] == 'F':

        k = 40

    elif row['tourney_level'] == 'M':

        k = 35

    else:

        k = 30

    

    elo_winner = elo(prev_elo_winner, exp_win, 1, row['score_diff'], k)

    elo_loser = elo(prev_elo_loser, exp_lose, 0, row['score_diff'], k)

    

    player_unique_list.loc[row['winner_id'],'elo_score'] = elo_winner

    player_unique_list.loc[row['loser_id'],'elo_score'] = elo_loser

    

    #return [elo_winner, elo_loser]



    return pd.Series({'elo_winner': elo_winner, 'elo_loser':elo_loser}) 
elo_df_gs = df[columns_needed_for_elo]

elo_df_gs['score_diff'] = elo_df_gs.apply(lambda x: score_diff(str(x['score']), x['best_of']), axis = 1)

elo_scores_gs = elo_df_gs.apply(lambda x: elo_calc_with_k(x), axis = 1)

elo_df_gs['elo_loser'] = elo_scores_gs.iloc[:,0]

elo_df_gs['elo_winner'] = elo_scores_gs.iloc[:,1]
gs_winner_list_5_or_more = grouped[grouped >= 5].index.values.tolist()
%matplotlib inline

import matplotlib.pyplot as plt



fig = plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(1,1,1)

ax.set_title('ELO Plot with Weighing Tournament Types')

ax.set_xlabel('Year')

ax.set_ylabel('ELO Score')

ax.set_ylim([1000, 2200])



for player in gs_winner_list_5_or_more:

    player_inst = elo_df_gs[elo_df_gs['winner_name'] == player]['elo_winner']

    player_inst = player_inst.groupby(level=0).mean()

    player_inst.plot(label= player)



plt.grid(b=True, which='major', color='k', linestyle='--', linewidth = .2)

plt.legend(loc=9, bbox_to_anchor=(1.15, 1))
%matplotlib inline

import matplotlib.pyplot as plt



fig = plt.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

ax = fig.add_subplot(1,1,1)

ax.set_title('ELO Plot with Weighing Tournament Types')

ax.set_xlabel('Year')

ax.set_ylabel('ELO Score')

ax.set_ylim([1000, 2200])



for player in gs_winner_list:

    player_inst = elo_df_gs[elo_df_gs['winner_name'] == player]['elo_winner']

    player_inst = player_inst.groupby(level=0).mean()

    player_inst.plot(label= player)



plt.grid(b=True, which='major', color='k', linestyle='--', linewidth = .2)

plt.legend(loc=9, bbox_to_anchor=(1.15, 1))
elo_df_gs.groupby('winner_name')['elo_winner'].max().sort_values(ascending = False).head(30)