# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
players = pd.read_csv('../input/Players.csv')

seasons_stats = pd.read_csv('../input/Seasons_Stats.csv')

player_data = pd.read_csv('../input/player_data.csv')
players.head()
len(players)
seasons_stats.head()
len(seasons_stats)
player_data.head()
len(player_data)
seasons_stats = seasons_stats[~seasons_stats.Player.isnull()]

players = players[~players.Player.isnull()]
players = players.rename(columns = {'Unnamed: 0':'id'})
num_players = player_data.groupby('name').count()

num_players =  num_players.iloc[:,:1]

num_players = num_players.reset_index()

num_players.columns = ['Player', 'count']

num_players[num_players['count'] > 1].head()
seasons_stats = seasons_stats.iloc[:,1:]

seasons_stats = seasons_stats.drop(['blanl', 'blank2'], axis=1)
player_data['id'] = player_data.index
mj_stats = seasons_stats[seasons_stats.Player == 'Michael Jordan*']

mj_stats['Year'].iloc[0] - mj_stats['Age'].iloc[0] 
seasons_stats['born'] = seasons_stats['Year'] - seasons_stats['Age'] - 1
players = players[~players.born.isnull()]
players_born = players[['Player', 'born']]
player_data = player_data[~player_data.birth_date.isnull()]
for i, row in player_data.iterrows():

    player_data.loc[i, 'born'] = float(row['birth_date'].split(',')[1])
player_data_born = player_data[['name', 'born']]

player_data_born.columns = ['Player', 'born']
born = pd.concat([players_born, player_data_born])
born = born.drop_duplicates()
born = born.reset_index()
born = born.drop('index', axis=1)
born['id'] = born.index
born[born.Player == 'Magic Johnson*']
seasons_stats[seasons_stats.Player == 'Magic Johnson*'].head(1)
born[born.Player == 'Hakeem Olajuwon*']
seasons_stats[seasons_stats.Player == 'Hakeem Olajuwon*'].head(1)
id_magic = born[born.Player == 'Magic Johnson*'].id.values[0]

id_hakeem = born[born.Player == 'Hakeem Olajuwon*'].id.values[0]

born.loc[id_magic, 'born'] = 1959

born.loc[id_hakeem, 'born'] = 1962
data = seasons_stats.merge(born, on=['Player', 'born'])
data = data[data.Tm != 'TOT']
# Filter players with at least 800 min in a season at played at least half of the matchs

data = data[(data.MP > 800) & (data.G > 40)]
# Per games

data['PPG'] = data['PTS'] / data['G']

data['APG'] = data['AST'] / data['G']

data['RPG'] = data['TRB'] / data['G']

data['SPG'] = data['STL'] / data['G']

data['BPG'] = data['BLK'] / data['G']

data['FPG'] = data['PF'] / data['G']

data['TOVPG'] = data['TOV'] / data['G']
# Adding mvps

mvp_players = {'Bob Pettit*': [1956, 1959],

                  'Bob Cousy*': [1957],

                  'Bill Russell*': [1958, 1961, 1962, 1963, 1965],

                  'Wilt Chamberlain*': [1960, 1966, 1967, 1968],

                  'Oscar Robertson*': [1964],

                  'Wes Unseld*': [1969],

                  'Willis Reed*': [1970],

                  'Kareem Abdul-Jabbar*': [1971, 1972, 1974, 1976, 1977, 1980],

                  'Dave Cowens*': [1973],

                  'Bob McAdoo*': [1975],

                  'Bill Walton*': [1978],

                  'Moses Malone*': [1979, 1982, 1983],

                  'Julius Erving*': [1981],

                  'Larry Bird*': [1984, 1985, 1986],

                  'Magic Johnson*': [1987, 1989, 1990],

                  'Michael Jordan*': [1988, 1991, 1992, 1996, 1998],

                  'Charles Barkley*': [1993],

                  'Hakeem Olajuwon*': [1994],

                  'David Robinson*': [1995],

                  'Karl Malone*': [1997, 1999],

                  'Shaquille O\'Neal*': [2000],

                  'Allen Iverson*': [2001],

                  'Tim Duncan': [2002, 2003],

                  'Kevin Garnett': [2004],

                  'Steve Nash': [2005, 2006],

                  'Dirk Nowitzki': [2007],

                  'Kobe Bryant': [2008],

                  'LeBron James': [2009, 2010, 2012, 2013],

                  'Derrick Rose': [2011],

                  'Kevin Durant': [2014],

                  'Stephen Curry': [2015, 2016],

                  'Russell Westbrook': [2017],

                  'James Harden': [2018]}
data['MVP'] = 0

for i, row in data.iterrows():  

    for k, v in mvp_players.items():

        for year in v:

            if row['Player'] != k:

                break

            elif(row['Year'] == year) & (row['Player'] == k):

                data.loc[i, 'MVP'] = 1

                break
data.columns
data = data[data.Year >= 2000]
# Adding Team Wins since 2000 to show this important paramater
data.sort_values(by='Tm').Tm.unique()
data[data.Tm == 'NOH'].Year.unique()
teams_wins = {'ATL': {2000:28, 2001:25, 2002:33, 2003:35, 2004:28, 2005:13, 2006:26, 2007:30, 2008:37, 2009:47, 2010:53, 2011:44, 2012:40, 2013:44, 2014:38, 2015:60, 2016:48, 2017:43},

             'BOS': {2000:35, 2001:36, 2002:49, 2003:44, 2004:36, 2005:45, 2006:33, 2007:24, 2008:66, 2009:62, 2010:50, 2011:56, 2012:39, 2013:41, 2014:25, 2015:40, 2016:48, 2017:53},

             'BRK': {2013:49, 2014:44, 2015:38, 2016:21, 2017:20},

              'CHA': {2005:18, 2006:26, 2007:33, 2008:32, 2009:35, 2010:44, 2011:34, 2012:7, 2013:21, 2014:43},

             'NJN': {2000:31, 2001:26, 2002:52, 2003:49, 2004:47, 2005:42, 2006:49, 2007:41, 2008:34, 2009:34, 2010:12, 2011:24, 2012:22},

             'CHH': {2000:49, 2001:46, 2002:44},

             'CHI': {2000:17, 2001:15, 2002:21, 2003:30, 2004:23, 2005:47, 2006:41, 2007:49, 2008:33, 2009:41, 2010:41, 2011:62, 2012:50, 2013:45, 2014:48, 2015:50, 2016:42, 2017:41},

             'CHO': {2015:33, 2016:48, 2017:36},

             'CLE': {2000:32, 2001:30, 2002:29, 2003:17, 2004:35, 2005:42, 2006:50, 2007:50, 2008:45, 2009:66, 2010:61, 2011:19, 2012:21, 2013:24, 2014:33, 2015:53, 2016:57, 2017:51},

             'DAL': {2000:40, 2001:53, 2002:57, 2003:60, 2004:52, 2005:58, 2006:60, 2007:67, 2008:51, 2009:50, 2010:55, 2011:57, 2012:36, 2013:41, 2014:49, 2015:50, 2016:42, 2017:33},

             'DEN': {2000:35, 2001:40, 2002:27, 2003:17, 2004:43, 2005:49, 2006:44, 2007:45, 2008:50, 2009:54, 2010:53, 2011:50, 2012:38, 2013:57, 2014:36, 2015:30, 2016:33, 2017:40},

             'DET': {2000:42, 2001:32, 2002:50, 2003:50, 2004:54, 2005:54, 2006:64, 2007:53, 2008:59, 2009:39, 2010:27, 2011:30, 2012:25, 2013:29, 2014:29, 2015:32, 2016:44, 2017:37},

             'GSW': {2000:19, 2001:17, 2002:21, 2003:38, 2004:37, 2005:34, 2006:34, 2007:42, 2008:48, 2009:29, 2010:26, 2011:36, 2012:23, 2013:47, 2014:51, 2015:67, 2016:73, 2017:67},

             'HOU': {2000:34, 2001:45, 2002:28, 2003:43, 2004:45, 2005:51, 2006:34, 2007:52, 2008:55, 2009:53, 2010:42, 2011:43, 2012:34, 2013:45, 2014:54, 2015:56, 2016:41, 2017:55},

             'IND': {2000:56, 2001:41, 2002:42, 2003:48, 2004:61, 2005:44, 2006:41, 2007:35, 2008:36, 2009:36, 2010:32, 2011:37, 2012:42, 2013:49, 2014:56, 2015:38, 2016:45, 2017:42},

             'LAC': {2000:15, 2001:31, 2002:39, 2003:27, 2004:28, 2005:37, 2006:47, 2007:40, 2008:23, 2009:19, 2010:29, 2011:32, 2012:40, 2013:56, 2014:57, 2015:56, 2016:53, 2017:51},

             'LAL': {2000:67, 2001:56, 2002:58, 2003:50, 2004:56, 2005:34, 2006:45, 2007:42, 2008:57, 2009:65, 2010:57, 2011:57, 2012:41, 2013:45, 2014:27, 2015:21, 2016:17, 2017:26},

             'MEM': {2002:23, 2003:28, 2004:50, 2005:45, 2006:49, 2007:22, 2008:22, 2009:24, 2010:40, 2011:46, 2012:41, 2013:56, 2014:50, 2015:55, 2016:42, 2017:43},

             'VAN': {2000:22, 2001:23},

              'MIA': {2000:52, 2001:50, 2002:36, 2003:25, 2004:42, 2005:59, 2006:52, 2007:44, 2008:15, 2009:43, 2010:47, 2011:58, 2012:46, 2013:66, 2014:54, 2015:37, 2016:48, 2017:41},

             'MIL': {2000:42, 2001:52, 2002:41, 2003:42, 2004:41, 2005:30, 2006:40, 2007:28, 2008:26, 2009:34, 2010:46, 2011:35, 2012:31, 2013:38, 2014:15, 2015:41, 2016:33, 2017:42},

             'MIN': {2000:50, 2001:47, 2002:50, 2003:51, 2004:58, 2005:44, 2006:33, 2007:32, 2008:22, 2009:24, 2010:15, 2011:17, 2012:26, 2013:31, 2014:40, 2015:16, 2016:29, 2017:31},

             'NOH': {2003:47, 2004:41, 2005:18, 2008:56, 2009:49, 2010:37, 2011:46, 2012:21, 2013:27},

             'NOK': {2006:38, 2007:39},

             'NOP': {2014:34, 2015:45, 2016:30, 2017:34},

             'NYK': {2000:50, 2001:48, 2002:30, 2003:37, 2004:39, 2005:33, 2006:23, 2007:33, 2008:23, 2009:32, 2010:29, 2011:42, 2012:36, 2013:54, 2014:37, 2015:17, 2016:32, 2017:31},

             'OKC': {2009:23, 2010:50, 2011:55, 2012:47, 2013:60, 2014:59, 2015:45, 2016:55, 2017:47},

             'ORL': {2000:41, 2001:43, 2002:44, 2003:42, 2004:21, 2005:36, 2006:36, 2007:40, 2008:52, 2009:59, 2010:59, 2011:52, 2012:37, 2013:20, 2014:23, 2015:25, 2016:35, 2017:29},

             'PHI': {2000:49, 2001:56, 2002:43, 2003:48, 2004:33, 2005:43, 2006:38, 2007:35, 2008:40, 2009:41, 2010:27, 2011:41, 2012:35, 2013:34, 2014:19, 2015:18, 2016:10, 2017:28},

             'PHO': {2000:53, 2001:51, 2002:36, 2003:44, 2004:29, 2005:62, 2006:54, 2007:61, 2008:55, 2009:46, 2010:54, 2011:40, 2012:33, 2013:25, 2014:48, 2015:39, 2016:23, 2017:24},

             'POR': {2000:59, 2001:50, 2002:49, 2003:50, 2004:41, 2005:27, 2006:21, 2007:32, 2008:41, 2009:54, 2010:50, 2011:48, 2012:28, 2013:33, 2014:54, 2015:51, 2016:44, 2017:41},

             'SAC': {2000:44, 2001:55, 2002:61, 2003:59, 2004:55, 2005:50, 2006:44, 2007:33, 2008:38, 2009:17, 2010:25, 2011:24, 2012:22, 2013:28, 2014:28, 2015:29, 2016:33, 2017:32},

             'SAS': {2000:53, 2001:58, 2002:58, 2003:60, 2004:57, 2005:59, 2006:63, 2007:58, 2008:56, 2009:54, 2010:50, 2011:61, 2012:50, 2013:58, 2014:62, 2015:55, 2016:67, 2017:61},

             'SEA': {2000:45, 2001:44, 2002:45, 2003:40, 2004:37, 2005:52, 2006:35, 2007:31, 2008:20},

             'TOR': {2000:45, 2001:47, 2002:42, 2003:24, 2004:33, 2005:33, 2006:27, 2007:47, 2008:41, 2009:33, 2010:40, 2011:22, 2012:23, 2013:34, 2014:48, 2015:49, 2016:56, 2017:51},

             'UTA': {2000:55, 2001:53, 2002:44, 2003:47, 2004:42, 2005:26, 2006:41, 2007:51, 2008:54, 2009:48, 2010:53, 2011:39, 2012:36, 2013:43, 2014:25, 2015:38, 2016:40, 2017:51},

             'WAS': {2000:29, 2001:19, 2002:37, 2003:37, 2004:25, 2005:45, 2006:42, 2007:41, 2008:43, 2009:19, 2010:26, 2011:23, 2012:20, 2013:29, 2014:44, 2015:46, 2016:41, 2017:49}}
for i, row in data.iterrows():  

    for k, v in teams_wins.items():

        for year, value in v.items():

            if ((row['Tm'] == k) & (row['Year'] == year)):

                data.loc[i, 'Tm_Wins'] = value
data_mvp = data[['id', 'Player', 'Year', 'PER', 'WS', 'BPM', 'VORP', 'PPG', 'Tm_Wins', 'MVP']]
data_mvp = data_mvp.fillna(0)
from sklearn.ensemble import RandomForestClassifier



years = range(2010, 2018)

mvp_years = dict()

results_mvp = pd.DataFrame(columns = ['id', 'Year', 'MVP'])



for y in years :

    # train : all seasons from 2000 to year

    # test : year

    train = data_mvp[data_mvp.Year < y]

    test = data_mvp[data_mvp.Year == y]

    X_train = train.drop(['id', 'Player', 'Year', 'MVP'], axis=1)

    y_train = train['MVP']

    X_test = test.drop(['id', 'Player', 'Year', 'MVP'], axis=1)

    

    # Random Forest



    random_forest = RandomForestClassifier(n_estimators=10)

    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)

    random_forest.score(X_train, y_train)

    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

    

    pred_proba = random_forest.predict_proba(X_test)

    

    y_pred_proba = []

    for i in enumerate(pred_proba):

        y_pred_proba.append(i[1][1])

    y_pred_proba = np.asarray(y_pred_proba)

    

    mvp_years = pd.DataFrame({

        "id": test["id"],

        "Year": y,

        "MVP": y_pred_proba

        })

    

    results_mvp = pd.concat([results_mvp, mvp_years])



results_mvp['id'] = results_mvp['id'].astype('int')

career_player = data[['id', 'Player']]

results_mvp = results_mvp.merge(career_player, on='id')



results_mvp = results_mvp.drop_duplicates()

# results_mvp = results_mvp.sort_values(by='MVP', ascending=False)

# results_mvp = results_mvp.iloc[0]
feature_importances = pd.DataFrame(random_forest.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances
top_mvp = results_mvp.sort_values('MVP', ascending=False).groupby('Year').head(1)

top_mvp = top_mvp.sort_values('Year', ascending=False)

top_mvp = top_mvp[['Year', 'Player']]

top_mvp