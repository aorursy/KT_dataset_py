import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)

%matplotlib inline
sns.set_style("whitegrid")
np.random.seed(42)

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def winner(row):
    if row['home_score'] > row['away_score']: return row['home_team'] 
    elif row['home_score'] < row['away_score']: return row['away_team']
    else: return 'DRAW'

def lose(row):
    if row['home_score'] > row['away_score']: return row['away_team'] 
    elif row['home_score'] < row['away_score']: return row['home_team']
    else: return 'DRAW'
data = pd.read_csv("../input/results.csv", parse_dates=['date'])
data.sample(5)
plt.figure(figsize=(20,10))
sns.barplot(data=pd.DataFrame(data.tournament.value_counts()[:10]).reset_index(),x='tournament', y='index', orient='h')
plt.xlabel("Count")
plt.xticks([i for i in range(500, 17000, 500)],rotation=90)
plt.ylabel("Type of Tournament")
plt.figure(figsize=(20,5))
sns.distplot(data[data['home_score']>0]['home_score'],kde=False,bins=30, color='g', label='Home Score')
sns.distplot(data[data['away_score']>0]['away_score'], kde=False, bins=30, color='r', label='Away Score')
plt.legend()
plt.xticks([i for i in range(1,21)])
plt.yticks([i for i in range(1000,13000,2000)])
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
major_world_tournaments = data[data['tournament'].isin(['Copa América','FIFA World Cup','UEFA Euro'])]
major_world_tournaments.sample(5)
colors=['lightskyblue','green']
plt.pie(major_world_tournaments['neutral'].value_counts(), labels=['tie','not a tie'], colors=colors,
       startangle=80, explode=(0.1,0), autopct='%1.1f%%', shadow=True)
win_lose = data
win_lose['winner'] = data.apply(lambda row: winner(row), axis=1)
win_lose['loser'] = win_lose.apply(lambda row: lose(row), axis=1)
win_lose.sample(5)
win_lose.pivot_table(index=['winner','loser'], aggfunc=[np.average, len], values=['home_score','away_score'])
