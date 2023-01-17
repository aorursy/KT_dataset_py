# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
games = pd.read_csv("../input/chess/games.csv")
games.head(2)
games['increment_code']
#Splitting the increment_code into two columns "Base_Time" and "Increment"

games['game_category'] = games['increment_code'].str.split('+').str[0]

games['increment'] = games['increment_code'].str.split('+').str[1]



games['game_category'] = games['game_category'].astype(int)

games['increment'] = games['increment'].astype(int)



#This approach is more like a short-cut and not a very generalized approach. This will be fixed soon.

games['game_category'][games['game_category'] >= 10] = 11

games['game_category'][games['game_category'] < 3] = 2

games['game_category'][(games['game_category'] >= 3) & (games['game_category'] < 10)] = 5



games['game_category'].replace(11, 'rapid', inplace = True)

games['game_category'].replace(2, 'bullet', inplace = True)

games['game_category'].replace(5, 'blitz', inplace = True)
games['game_category'].value_counts()
games['game_category'].unique()
pd.cut(games['turns'], 10, precision=0).value_counts().sort_index().plot.bar()
#games['victory_status'].value_counts().plot.bar()

ax = sns.countplot( x = 'winner', hue = 'victory_status', data = games)

ax.set(xlabel = 'Who Won?')
#Distribution of white's rating

#pd.cut(games['white_rating'], 20, precision = 0).value_counts().sort_index().plot.bar()

sns.distplot(games['white_rating'], bins = 30)



#average rating of white

print('Average Rating of White is:',games['white_rating'].mean())

#Distribution of black's rating

#pd.cut(games['black_rating'], 20, precision = 0).value_counts().sort_index().plot.bar()

sns.distplot(games['black_rating'], bins = 30)



#average rating of black

print('Average rating of Black is:',games['black_rating'].mean())
#plotting black and white together to compare the scale of their distribution

sns.distplot(games['black_rating'], bins = 30, color = 'r')

sns.distplot(games['white_rating'], bins = 30, color = 'b')
#games['winner'].value_counts().plot.bar()

ax = sns.countplot( x = games['winner'], data = games)

ax.set(xlabel = 'Who Won?')



black_black = games['black_rating'][games['winner'] == 'black'].mean()

black_white = games['black_rating'][games['winner'] == 'white'].mean()

white_black = games['white_rating'][games['winner'] == 'black'].mean()

white_white = games['white_rating'][games['winner'] == 'white'].mean()

print("The average rating of Black when Black won =", black_black)

print("The average rating of Black when Black lost =", black_white)

print("The average rating of White when White won =", white_white)

print("The average rating of White when White lost=", white_black)
#adding a new column which stores the rating difference between black and white



games["Rating_Diff"] = abs(games['white_rating'] - games['black_rating'])
#plotting the distribution of rating difference

#pd.cut(games["Rating_Diff"], 30, precision = 0).value_counts().sort_index().plot.bar()

sns.distplot(games["Rating_Diff"], bins = 30)
#plotting the rating difference distribution when white won

pd.cut(games["Rating_Diff"][games['winner'] == 'white'], 30, precision = 0).value_counts().sort_index().plot.bar()
#plotting the rating difference distribution when black won

pd.cut(games["Rating_Diff"][games['winner'] == 'black'], 30, precision = 0).value_counts().sort_index().plot.bar()
#plotting the Rating_Diff distribution for white and black together to compare the scales of both and gain some inference

sns.distplot(games['Rating_Diff'][games['winner'] == 'black'], color = 'red')

sns.distplot(games['Rating_Diff'][games['winner'] == 'white'], color = 'blue')
opening_dist = games['opening_name'].value_counts()



#plotting the top 30 opening played

#opening_dist[:30].plot.bar()



#sns.countplot(games['opening_name'])

chart = sns.barplot( x = opening_dist.index[:10], y = opening_dist[:10])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
#count of openings which lead to a win for black

winning_openings_black = games['opening_name'][games['winner'] == 'black'].value_counts()



#plotting the top 10 openings which lead to a win for black/lead to a loss for white

#winning_openings_black[:10].plot.bar()



chart = sns.barplot( x = winning_openings_black.index[:10], y = winning_openings_black[:10])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
#count of openings which lead to a win for white

winning_openings_white = games['opening_name'][games['winner'] == 'white'].value_counts()



#plotting top 10 openings which lead to a win for white/lead to a loss for black

#winning_openings_white[:10].plot.bar()

chart = sns.barplot( x = winning_openings_white.index[:10], y = winning_openings_white[:10])

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
#Collating the top 10 openings for white and black

winning_openings_white_top10 = winning_openings_white[:10]

winning_openings_black_top10 = winning_openings_black[:10]



#Getting the Data of the Top 10 Openings for White and Black in a separate DataFrame for each

openingSetWhite = games[games['opening_name'].isin(winning_openings_white_top10.index)]

openingSetBlack = games[games['opening_name'].isin(winning_openings_black_top10.index)]



#Getting the count of each opening for White in 'total_count_white'

# & Getting the count of each opening for White where White is the winner in 'total_count_white_winner'

total_count_white = openingSetWhite.groupby('opening_name').count()

total_count_white_winner = openingSetWhite[openingSetWhite['winner'] == 'white'].groupby('opening_name').count()



#Getting the count of each opening for Black in 'total_count_black'

# & Getting the count of each opening for Black where Black is the winner in 'total_count_black_winner'

total_count_black = openingSetBlack.groupby('opening_name').count()

total_count_black_winner = openingSetBlack[openingSetBlack['winner'] == 'black'].groupby('opening_name').count()



#Calculating winning percentage for the top 10 openings for Black and White each

winning_perc_white = (total_count_white_winner/total_count_white)*100

winning_perc_black = (total_count_black_winner/total_count_black)*100
ax = sns.barplot(x=winning_perc_white.index , y=winning_perc_white.id)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set(title = 'Top 10 Openings for White with the Highest Winning %')
ax = sns.barplot(x=winning_perc_black.index , y=winning_perc_black.id)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set(title = 'Top 10 Openings for Black with the Highest Winning %')
sns.countplot( x = 'game_category', hue = 'winner', data = games)
sns.countplot( x = 'game_category', hue = 'victory_status', data = games)
hue_order = games['victory_status'].unique()
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(1,3,figsize=a4_dims)

#ax.set(xlabel = 'Time Format', ylabel = 'Count', title = 'Victory Status ')

sns.countplot( x = 'game_category', hue = 'victory_status', data = games[games['game_category'] == 'rapid'], hue_order = hue_order, ax = ax[0])

sns.countplot( x = 'game_category', hue = 'victory_status', data = games[games['game_category'] == 'blitz'], hue_order = hue_order, ax = ax[1])

sns.countplot( x = 'game_category', hue = 'victory_status', data = games[games['game_category'] == 'bullet'], hue_order = hue_order, ax = ax[2])
games.head(2)
games['Rating_Diff_Sign'] = games['white_rating'] - games['black_rating']

games['Sign'] = ''

bool_pos = games['Rating_Diff_Sign'][games['Rating_Diff_Sign'] > 0]
topWhitePlayers = games['white_id'][games['winner'] == 'white'].value_counts()

topBlackPlayers = games['black_id'][games['winner'] == 'black'].value_counts()
df = pd.DataFrame(topWhitePlayers.append(topBlackPlayers), columns = ['Matches_Won'])

df.sort_values(by = ['Matches_Won'], ascending = False, inplace = True)
df['Player_ID'] = df.index
top_players = df.groupby('Player_ID').sum().reset_index()

top_players.sort_values('Matches_Won', ascending=False, inplace = True)
top_players[:10]