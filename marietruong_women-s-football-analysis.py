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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv', index_col = 'date', parse_dates = True)
df
plt.figure(figsize = (20,10))
plt.title('Number of games played')
ax = sns.barplot(x =df.groupby(df.index.year).count().index ,y = df.groupby(df.index.year).count().home_team)
ax.set(ylabel = 'Games played')
plt.show()
plt.figure(figsize = (20,10))
plt.title('Top Home Scorers in 2019')
sns.barplot(x =df['2019'].groupby('home_team').sum().sort_values('home_score', ascending = False).index[0:10], y =df['2019'].groupby('home_team').sum().home_score.sort_values(ascending = False)[0:10]  )
plt.figure(figsize = (20,10))
plt.title('Top Away Scorers in 2019')
sns.barplot(x =df['2019'].groupby('away_team').sum().sort_values('away_score', ascending = False).index[0:10], y =df['2019'].groupby('away_team').sum().away_score.sort_values(ascending = False)[0:10]  )
home_scorers = df['2019'].groupby('home_team').sum().sort_values('home_score', ascending = False)
away_scorers = df['2019'].groupby('away_team').sum().sort_values('away_score', ascending = False)
goals = home_scorers.merge(away_scorers, how = 'outer', left_index = True, right_index = True)
goals = goals [['home_score_x', 'away_score_y']]
goals = goals.rename(columns = {'home_score_x' : 'home', 'away_score_y' : 'away'})
goals = goals.fillna(0, axis = 0)
goals['total'] = goals['home']+ goals['away']
plt.figure(figsize = (20,10))
plt.title('Top scorers in 2019')
sns.barplot(x = goals.sort_values('total', ascending = False).index[0:10], y = goals.sort_values('total', ascending = False).total[0:10])
alltime_home = df.groupby('home_team').mean().sort_values('home_score', ascending = False)
alltime_away = df.groupby('away_team').mean().sort_values('away_score', ascending = False)
alltime_goals = alltime_home.merge(alltime_away, how = 'outer', left_index = True, right_index = True)
alltime_goals = alltime_goals.fillna(0, axis = 1)
alltime_goals = alltime_goals[['home_score_x', 'away_score_x', 'home_score_y', 'away_score_y']]
alltime_goals['Goals scored'] = 0
alltime_goals['Goals scored'] = (alltime_goals['home_score_x'] + alltime_goals['away_score_y'])/2
alltime_goals['Goals taken'] = (alltime_goals['away_score_x'] + alltime_goals['home_score_y'])/2
alltime_goals = alltime_goals[['Goals scored', 'Goals taken']]
plt.figure(figsize = (20,10))
plt.title('Best Average Scorers')
sns.barplot(x = alltime_goals.sort_values('Goals scored', ascending = False).index[0:15], y =alltime_goals.sort_values('Goals scored', ascending = False)['Goals scored'][0:15]) 

plt.figure(figsize = (20,10))
plt.title('Best Average Defense')
sns.barplot(x = alltime_goals.sort_values('Goals taken', ascending = True).index[0:15], y =alltime_goals.sort_values('Goals taken', ascending = True)['Goals taken'][0:15]) 


df['Goals'] = df.home_score + df.away_score
plt.figure()
plt.title('Average number of goals per game')
sns.lineplot(x= df.groupby(df.index.year).mean().index, y = df.groupby(df.index.year).mean().Goals)
plt.figure(figsize = (20,10))
plt.title('Countries that have hosted the more games')
plt.ylabel('Number of games hosted')
ax = sns.barplot(x = df.groupby('country').count().sort_values('home_team', ascending = False).index[0:10], y = df.groupby('country').count().sort_values('home_team', ascending = False).home_team[0:10])
ax.set(ylabel = 'Games hosted')
plt.show()
France = df[(df.home_team == 'France') | (df.away_team == 'France')]
plt.figure(figsize = (20,10))
plt.title('Number of games played by France')
ax = sns.barplot(x =France.groupby(France.index.year).count().index ,y = France.groupby(France.index.year).count().home_team)
ax.set(ylabel = 'Games played')
plt.show()
France['2020']
home = France[France.home_team == 'France']
away = France[France.home_team != 'France']
plt.figure(figsize = (15,8))
plt.title('Goals scored at home')
sns.lineplot(y= home.groupby(home.index.year).mean().home_score, x = home.groupby(home.index.year).mean().index, label = 'Home')
sns.lineplot(y= away.groupby(away.index.year).mean().away_score, x = away.groupby(away.index.year).mean().index, label = 'Away')
France['diff_score'] = France.home_score - France.away_score
France['Issue'] = 0
France['Issue'][(France.diff_score >0) & (France.home_team == 'France')] = 'Victory'
France['Issue'][(France.diff_score >0) & (France.home_team != 'France')] = 'Defeat'
France['Issue'][(France.diff_score <0) & (France.home_team == 'France')] = 'Defeat'
France['Issue'][(France.diff_score <0) & (France.home_team != 'France')] = 'Victory'
France['Issue'][France.diff_score  == 0] = 'Null'
plt.figure()
ax = sns.barplot(x = France.groupby('Issue').count().index, y = France.groupby('Issue').count().home_team)
ax.set(ylabel ='Total number')
plt.show()

victories = France[France.Issue == 'Victory']
defeats = France[France.Issue == 'Defeat']
nulls = France[France.Issue == 'Null']
plt.figure(figsize = (20,10))
plt.title('French Team Results')
sns.lineplot(x = victories.groupby(victories.index.year).count().index, y = victories.groupby(victories.index.year).count().home_team, label = 'Victories', c = 'green')
sns.lineplot(x = defeats.groupby(defeats.index.year).count().index, y = defeats.groupby(defeats.index.year).count().home_team, label = 'Defeats', c = 'red')
sns.lineplot(x = nulls.groupby(nulls.index.year).count().index, y = nulls.groupby(nulls.index.year).count().home_team, label = 'Nulls', c = 'orange')
plt.show()
France
France['Home'] = 0
France['Home'][France['home_team'] == 'France'] = 1
France['Opponent'] = 0
France['Opponent'][France['Home'] == 1] = France['away_team']
France['Opponent'][France['Home'] == 0] = France['home_team']
France['France_Score'] = 0
France['France_Score'][France['Home'] == 0]= France['away_score']
France['France_Score'][France['Home'] == 1]= France['home_score']
France['Opponent_Score'] = 0
France['Opponent_Score'][France['Home'] == 1]= France['away_score']
France['Opponent_Score'][France['Home'] == 0]= France['home_score']
France = France[['tournament', 'city', 'country', 'neutral', 'Goals', 'Home', 'Opponent', 'France_Score', 'Opponent_Score', 'Issue']]
France
plt.figure(figsize = (20,10))
plt.title('France wins against ...')
ax = sns.barplot(x = France[France.Issue == 'Victory'].groupby('Opponent').count().sort_values('Issue', ascending = False).index[0:10], y = France[France.Issue == 'Victory'].groupby('Opponent').count().tournament.sort_values(ascending = False)[0:10] )
ax.set(ylabel = 'Games won')
plt.show()
plt.figure(figsize = (20,10))
plt.title('France loses against ...')
ax = sns.barplot(x = France[France.Issue == 'Defeat'].groupby('Opponent').count().sort_values('Issue', ascending = False).index[0:10], y = France[France.Issue == 'Defeat'].groupby('Opponent').count().tournament.sort_values(ascending = False)[0:10] )
ax.set(ylabel = 'Games lost')
plt.show()
plt.figure(figsize = (20,10))
plt.title('France struggles against ...')
ax = sns.barplot(x = France[France.Issue == 'Null'].groupby('Opponent').count().sort_values('Issue', ascending = False).index[0:10], y = France[France.Issue == 'Null'].groupby('Opponent').count().tournament.sort_values(ascending = False)[0:10] )
ax.set(ylabel = 'Null games')
plt.show()
plt.figure(figsize = (20,10))
plt.title('France scores best against ...')
ax = sns.barplot(x = France.groupby('Opponent').mean().sort_values('France_Score', ascending = False).index[0:10], y = France.groupby('Opponent').mean().France_Score.sort_values(ascending = False)[0:10] )
ax.set(ylabel = 'Average goals scored by France')
plt.show()
plt.figure(figsize = (20,10))
plt.title('France defense has a hard time against ...')
ax = sns.barplot(x = France.groupby('Opponent').mean().sort_values('Opponent_Score', ascending = False).index[0:10], y = France.groupby('Opponent').mean().Opponent_Score.sort_values(ascending = False)[0:10] )
ax.set(ylabel = 'Average goals scored by opponent')
plt.show()
plt.figure(figsize = (20,10))
plt.title('France scores best in ...')
ax = sns.barplot(x = France.groupby('tournament').mean().sort_values('France_Score', ascending = False).index, y = France.groupby('tournament').mean().France_Score.sort_values(ascending = False) )
ax.set(ylabel = 'Average goals scored by France')
plt.show()
plt.figure(figsize = (20,10))
plt.title('France takes a lot of goals  in ...')
ax = sns.barplot(x = France.groupby('tournament').mean().sort_values('Opponent_Score', ascending = False).index, y = France.groupby('tournament').mean().Opponent_Score.sort_values(ascending = False) )
ax.set(ylabel = 'Average goals scored by opponent')
plt.show()