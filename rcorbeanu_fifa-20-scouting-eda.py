# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import itertools 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')
df.head()
# dropping empty columns

df.dropna(axis=1, how='all')
df.isnull().sum()
df.drop(['sofifa_id','player_url','real_face','nation_position','nation_jersey_number',],axis=1,inplace=True)
df.columns
df[['age','height_cm','weight_kg','overall']].describe()
promising_youngsters = df.loc[(df['overall']>=70) & (df['age']<=21)]

promising_youngsters.head()
promising_youngsters['club'].value_counts().head(20)
promising_youngsters['nationality'].value_counts().head(20)
England = len(promising_youngsters[promising_youngsters['nationality'] == 'England'])

Germany = len(promising_youngsters[promising_youngsters['nationality'] == 'Germany'])

Spain = len(promising_youngsters[promising_youngsters['nationality'] == 'Spain'])

Argentina = len(promising_youngsters[promising_youngsters['nationality'] == 'Argentina'])

France = len(promising_youngsters[promising_youngsters['nationality'] == 'France'])

Brazil = len(promising_youngsters[promising_youngsters['nationality'] == 'Brazil'])

Italy = len(promising_youngsters[promising_youngsters['nationality'] == 'Italy'])

Colombia = len(promising_youngsters[promising_youngsters['nationality'] == 'Colombia'])

Japan = len(promising_youngsters[promising_youngsters['nationality'] == 'Japan'])

Netherlands = len(promising_youngsters[promising_youngsters['nationality'] == 'Netherlands'])



labels = 'England','Germany','Spain','Argentina','France','Brazil','Italy','Colombia','Japan','Netherlands'

sizes = [England,Germany,Spain,Argentina,France,Brazil,Italy,Colombia,Japan,Netherlands]

plt.figure(figsize=(6,6))

col=['red','blue','green','orange','purple','brown','black','yellow','pink','magenta']



plt.pie(sizes, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), labels=labels, colors=col,

autopct='%1.1f%%', shadow=True, startangle=90)

plt.title('Ratio of promising youngsters by nationality among selected nations', fontsize=16)

plt.show()
all_clubs = df['club'].unique() 

all_clubs.tolist() 



la_liga = ['Alavés', 'Atlético Madrid', 'FC Barcelona', 'RC Celta', 'SD Eibar', 'RCD Espanyol', 'Real Madrid', 'Valencia CF',

        'Real Betis', 'Getafe CF', 'Villarreal CF', 'Sevilla FC', 'Real Sociedad', 'Athletic Club de Bilbao', 'Deportivo Alavés',

        'Levante UD', 'Real Valladolid CF', 'CD Leganés', 'Granada CF', 'CA Osasuna']

premier_league = ['Arsenal', 'Manchester City', 'Liverpool', 'Tottenham Hotspur', 'Manchester United', 'Chelsea', 'West Ham United',

          'Wolverhampton Wanderers', 'Everton', 'Crystal Palace', 'Leicester City', 'Watford', 'Bournemouth', 'Newcastle United',

          'Brighton & Hove Albion', 'Norwich City', 'Aston Villa', 'Southampton', 'Burnley', 'Sheffield United']

bundesliga =['FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen', 'SV Werder Bremen', 'Borussia Mönchengladbach',

         'VfL Wolfsburg', 'Eintracht Frankfurt', 'Hertha BSC', 'TSG 1899 Hoffenheim', '1. FC Köln', 'FC Schalke 04', '1. FSV Mainz 05', 

         'FC Augsburg', 'SC Freiburg', 'Hannover 96', 'VfB Stuttgart', 'Fortuna Düsseldorf', '1. FC Union Berlin', 'SC Paderborn 07']

ligue_1 = ['AS Saint-Étienne', 'Paris Saint-Germain', 'Olympique Lyonnais', 'Olympique de Marseille', 'AS Monaco', 'Montpellier HSC', 

         'FC Girondins de Bordeaux', 'LOSC Lille', 'FC Nantes', 'OGC Nice', 'RC Strasbourg Alsace', 'Toulouse Football Club', 'Angers SCO',

         'Stade Rennais FC', 'Nîmes Olympique', 'Amiens SC', 'En Avant de Guingamp', 'Dijon FCO', 'Stade Brestois 29', 'FC Metz']

serie_a = ['Juventus', 'Napoli', 'Inter', 'Lazio', 'Milan', 'Atalanta', 'Torino', 'Roma', 'Sampdoria', 'Cagliari', 'Brescia', 'Sassuolo',

        'Fiorentina', 'Genoa', 'Udinese', 'Parma', 'Bologna', 'SPAL', 'Hellas Verona', 'Lecce']



big_5 = itertools.chain(la_liga, premier_league, bundesliga, ligue_1, serie_a)

big_5_leagues = list(big_5)



rest_of_world = [x for x in all_clubs if x not in big_5_leagues] 
# filtering the dataframe by each of the big 5 leagues as well as everything else in the rest of world 

laliga_df = df[df['club'].isin(la_liga)]

premierleague_df = df[df['club'].isin(premier_league)]

bundesliga_df = df[df['club'].isin(bundesliga)]

ligue1_df = df[df['club'].isin(ligue_1)]

seriea_df = df[df['club'].isin(serie_a)]

restofworld_df = df[df['club'].isin(rest_of_world)]



# filtering the above for promising youngsters

laliga_youth = laliga_df.loc[(laliga_df['overall']>=70) & (laliga_df['age']<=21)]

premierleague_youth = premierleague_df.loc[(premierleague_df['overall']>=70) & (premierleague_df['age']<=21)]

bundesliga_youth = bundesliga_df.loc[(bundesliga_df['overall']>=70) & (bundesliga_df['age']<=21)]

ligue1_youth = ligue1_df.loc[(ligue1_df['overall']>=70) & (ligue1_df['age']<=21)]

seriea_youth = seriea_df.loc[(seriea_df['overall']>=70) & (seriea_df['age']<=21)]

restofworld_youth = restofworld_df.loc[(restofworld_df['overall']>=70) & (restofworld_df['age']<=21)]
# Calculating total number of youngsters in each of the big 5 leagues that are rated 70 and over

print('Total number of U21 players in La Liga with a minimum rating of 70: {}'.format(laliga_youth['long_name'].count()))

print('Total number of U21 players in Premier League with a minimum rating of 70: {}'.format(premierleague_youth['long_name'].count()))

print('Total number of U21 players in Bundesliga with a minimum rating of 70: {}'.format(bundesliga_youth['long_name'].count()))

print('Total number of U21 players in Ligue 1 with a minimum rating of 70: {}'.format(ligue1_youth['long_name'].count()))

print('Total number of U21 players in Serie A with a minimum rating of 70: {}'.format(seriea_youth['long_name'].count()))

print('Total number of U21 players in Rest of World with a minimum rating of 70: {}'.format(restofworld_youth['long_name'].count()))
# bar graph chart comparing the total number youngsters in the 5 big leagues with a minimum rating of 70

x = ['La Liga', 'EPL', 'Bundesliga', 'Ligue 1', 'Serie A']

y = [37, 44, 56, 31, 39]



plt.bar(x, height=y, width=0.8, color='red')

plt.title('Total number of U21 players in each of the big 5 leagues (minimum rating of 70)', fontsize=16)

plt.show()
# analyzing promising youngsters by club from the big 5 leagues 

ll = laliga_youth['club'].value_counts()

pl = premierleague_youth['club'].value_counts()

bl = bundesliga_youth['club'].value_counts()

l = ligue1_youth['club'].value_counts()

sa = seriea_youth['club'].value_counts()



a = ll[::-1].index.tolist()

b = pl[::-1].index.tolist()

c = bl[::-1].index.tolist()

d = l[::-1].index.tolist()

e = sa[::-1].index.tolist()



plt.plot(a, ll[::-1], b, pl[::-1], c, bl[::-1], d, l[::-1], e, sa[::-1])

plt.tick_params(axis='x', which='major', labelsize=1)

plt.title('Promising Youngsters By Club')

plt.xlabel('Clubs')

plt.ylabel('# of U21 min-70 Rated Players')

labels = ['La Liga', 'Premier League', 'Bundesliga', 'Ligue 1', 'Serie A']

plt.legend(labels)

plt.show()
# analyzing the rest of world and grouping by club to see which clubs appear the most

other_talent = restofworld_youth['club'].value_counts()

other_values_list = other_talent.values.tolist()

other_index_list = other_talent.index.tolist()



# plt.bar(other_index_list[:10], other_values_list[:10], width=0.8, color='blue')

# plt.figure(figsize=(100, 10))

# plt.xticks(other_index_list[:10], other_index_list[:10], rotation='vertical')

# plt.show()

plt.bar(other_index_list[:5], other_values_list[:5], width = 0.8, color = 'blue')

plt.title('Talent Hodbeds Outside the Top 5 Leagues')

plt.ylabel('# of U21 players rated 70 or more')

plt.xlabel('Club')