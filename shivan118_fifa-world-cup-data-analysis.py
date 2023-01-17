from IPython.display import Image

Image(filename='/kaggle/input/fifaworld/1_0A8eTfcCEI4vQdErHdrwEQ.jpeg', width="800", height='50')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly as py

import cufflinks as cf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
players = pd.read_csv("/kaggle/input/fifa-world-cup/WorldCupPlayers.csv")

matches = pd.read_csv("/kaggle/input/fifa-world-cup/WorldCupMatches.csv")

world_cup = pd.read_csv("/kaggle/input/fifa-world-cup/WorldCups.csv")
players.head()
matches.head()
matches.tail()
world_cup.head()
matches.dropna(subset=['Year'], inplace=True)
matches.tail()
matches['Home Team Name'].value_counts()
names = matches[matches['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()

names
wrong = list(names.index)

wrong
correct = [name.split('>')[1] for name in wrong]

correct
old_name = ['Germany FR', 'Maracan� - Est�dio Jornalista M�rio Filho', 'Estadio do Maracana']

new_name = ['Germany', 'Maracan Stadium', 'Maracan Stadium']
wrong = wrong + old_name

correct = correct + new_name
wrong, correct
for index, wr in enumerate(wrong):

    world_cup = world_cup.replace(wrong[index], correct[index])

    

for index, wr in enumerate(wrong):

    matches = matches.replace(wrong[index], correct[index])

    

for index, wr in enumerate(wrong):

    players = players.replace(wrong[index], correct[index])
names = matches[matches['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()

names
winner = world_cup['Winner'].value_counts()

winner
runnerup = world_cup['Runners-Up'].value_counts()

runnerup
third = world_cup['Third'].value_counts()

third
teams = pd.concat([winner, runnerup, third], axis=1)

teams.fillna(0, inplace=True)

teams = teams.astype(int)

teams
from plotly.offline import iplot

py.offline.init_notebook_mode(connected=True)

cf.go_offline()
teams.iplot(kind = 'bar', xTitle='Teams', yTitle='Count', title='FIFA World Cup Winning Count')
matches.head(2)
home = matches[['Home Team Name', 'Home Team Goals']].dropna()

away = matches[['Away Team Name', 'Away Team Goals']].dropna()
home.columns = ['Countries', 'Goals']

away.columns = home.columns
goals = home.append(away, ignore_index = True)
goals = goals.groupby('Countries').sum()

goals
goals = goals.sort_values(by = 'Goals', ascending=False)

goals
goals[:20].iplot(kind='bar', xTitle = 'Country Names', yTitle = 'Goals', title = 'Countries Hits Number of Goals')
world_cup['Attendance'] = world_cup['Attendance'].str.replace(".", "")
world_cup.head()
fig, ax = plt.subplots(figsize = (10,5))

sns.despine(right = True)

g = sns.barplot(x = 'Year', y = 'Attendance', data = world_cup)

g.set_xticklabels(g.get_xticklabels(), rotation = 80)

g.set_title('Attendance Per Year')



#==========================================



fig, ax = plt.subplots(figsize = (10,5))

sns.despine(right = True)

g = sns.barplot(x = 'Year', y = 'QualifiedTeams', data = world_cup)

g.set_xticklabels(g.get_xticklabels(), rotation = 80)

g.set_title('Qualified Teams Per Year')



#==========================================



fig, ax = plt.subplots(figsize = (10,5))

sns.despine(right = True)

g = sns.barplot(x = 'Year', y = 'GoalsScored', data = world_cup)

g.set_xticklabels(g.get_xticklabels(), rotation = 80)

g.set_title('Goals Scored by Teams Per Year')





#==========================================





fig, ax = plt.subplots(figsize = (10,5))

sns.despine(right = True)

g = sns.barplot(x = 'Year', y = 'MatchesPlayed', data = world_cup)

g.set_xticklabels(g.get_xticklabels(), rotation = 80)

g.set_title('Matches Plyed Scored by Teams Per Year')



matches.head(2)
home = matches.groupby(['Year', 'Home Team Name'])['Home Team Goals'].sum()

home
away = matches.groupby(['Year', 'Away Team Name'])['Away Team Goals'].sum()

away
goals = pd.concat([home, away], axis=1)

goals.fillna(0, inplace=True)

goals['Goals'] = goals['Home Team Goals'] + goals['Away Team Goals']

goals = goals.drop(labels = ['Home Team Goals', 'Away Team Goals'], axis = 1)

goals
goals = goals.reset_index()
goals.columns = ['Year', 'Country', 'Goals']

goals = goals.sort_values(by = ['Year', 'Goals'], ascending = [True, False])

goals
top5 = goals.groupby('Year').head()

top5.head(10)
import plotly.graph_objects as go


x, y = goals['Year'].values, goals['Goals'].values
data = []

for team in top5['Country'].drop_duplicates().values:

    year = top5[top5['Country'] == team]['Year']

    goal = top5[top5['Country'] == team]['Goals']

    

    data.append(go.Bar(x = year, y = goal, name = team))

layout = go.Layout(barmode = 'stack', title = 'Top 5 Teams with most Goals', showlegend = False)



fig = go.Figure(data = data, layout = layout)

fig.show()
matches['Datetime'] = pd.to_datetime(matches['Datetime'])
matches['Datetime'] = matches['Datetime'].apply(lambda x: x.strftime('%d %b, %y'))
top10 = matches.sort_values(by = 'Attendance', ascending = False)[:10]

top10['vs'] = top10['Home Team Name'] + " vs " + top10['Away Team Name']



plt.figure(figsize = (12,10))



ax = sns.barplot(y = top10['vs'], x = top10['Attendance'])

sns.despine(right = True)



plt.ylabel('Match Teams')

plt.xlabel('Attendence')

plt.title('Matches with the highest number of Attendence')



for i, s in enumerate("Stadium: " + top10['Stadium'] +", Date: " + top10['Datetime']):

    ax.text(2000, i, s, fontsize = 12, color = 'white')

plt.show()
matches['Year'] = matches['Year'].astype(int)



std = matches.groupby(['Stadium', 'City'])['Attendance'].mean().reset_index().sort_values(by = 'Attendance', ascending =False)



top10 = std[:10]



plt.figure(figsize = (12,9))

ax = sns.barplot(y = top10['Stadium'], x = top10['Attendance'])

sns.despine(right = True)



plt.ylabel('Stadium Names')

plt.xlabel('Attendance')

plt.title('Stadium with the heighest number of attendance')

for i, s in enumerate("City: " + top10['City']):

        ax.text(2000, i, s, fontsize = 12, color = 'b')

        

plt.show()
matches['City'].value_counts()[:20].iplot(kind = 'bar')
gold = world_cup["Winner"]

silver = world_cup["Runners-Up"]

bronze = world_cup["Third"]



gold_count = pd.DataFrame.from_dict(gold.value_counts())

silver_count = pd.DataFrame.from_dict(silver.value_counts())

bronze_count = pd.DataFrame.from_dict(bronze.value_counts())

podium_count = gold_count.join(silver_count, how='outer').join(bronze_count, how='outer')

podium_count = podium_count.fillna(0)

podium_count.columns = ['WINNER', 'SECOND', 'THIRD']

podium_count = podium_count.astype('int64')

podium_count = podium_count.sort_values(by=['WINNER', 'SECOND', 'THIRD'], ascending=False)



podium_count.plot(y=['WINNER', 'SECOND', 'THIRD'], kind="bar", 

                  color =['gold','silver','brown'], figsize=(15, 6), fontsize=14,

                 width=0.8, align='center')

plt.xlabel('Countries')

plt.ylabel('Number of podium')

plt.title('Number of podium by country')
#world_cups_matches['Win conditions'].value_counts()

home = matches[['Home Team Name', 'Home Team Goals']].dropna()

away = matches[['Away Team Name', 'Away Team Goals']].dropna()



goal_per_country = pd.DataFrame(columns=['countries', 'goals'])

goal_per_country = goal_per_country.append(home.rename(index=str, columns={'Home Team Name': 'countries', 'Home Team Goals': 'goals'}))

goal_per_country = goal_per_country.append(away.rename(index=str, columns={'Away Team Name': 'countries', 'Away Team Goals': 'goals'}))



goal_per_country['goals'] = goal_per_country['goals'].astype('int64')



goal_per_country = goal_per_country.groupby(['countries'])['goals'].sum().sort_values(ascending=False)



goal_per_country[:10].plot(x=goal_per_country.index, y=goal_per_country.values, kind="bar", figsize=(12, 6), fontsize=14)

plt.xlabel('Countries')

plt.ylabel('Number of goals')

plt.title('Top 10 of Number of goals by country')


def get_labels(matches):

    if matches['Home Team Goals'] > matches['Away Team Goals']:

        return 'Home Team Win'

    if matches['Home Team Goals'] < matches['Away Team Goals']:

        return 'Away Team Win'

    return 'DRAW'
matches['outcome'] = matches.apply(lambda x: get_labels(x), axis=1)
matches.head()
mt = matches['outcome'].value_counts()

mt
plt.figure(figsize = (6,6))



mt.plot.pie(autopct = "%1.0f%%", colors = sns.color_palette('winter_r'), shadow = True)



c = plt.Circle((0,0), 0.4, color =  'white')

plt.gca().add_artist(c)

plt.title('Match Outcomes by Home and Away Teams')

plt.show()