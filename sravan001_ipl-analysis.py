# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_1 = pd.read_csv('/kaggle/input/ipl/matches.csv')

data_2 = pd.read_csv('/kaggle/input/ipl/deliveries.csv') 

data_1.head()
data_1.shape
data_1.isnull().sum()
data_1.drop(columns='umpire3',inplace=True)
data_1.isnull().sum()
def nans(df): return df[df.isnull().any(axis=1)]

nans(data_1)
nans(data_1)
data_1.city.fillna('Dubai',inplace=True)
data_1.umpire1.fillna('Anonymous',inplace=True)

data_1.umpire2.fillna('Anonymous',inplace=True)
data_1.isnull().sum()
nans(data_1)
data_1.winner.fillna('Match Abandoned',inplace=True)

data_1.player_of_match.fillna('Match Abandoned',inplace=True)
data_1.isnull().sum()
data_1.info()
data_1['date'] = pd.to_datetime(data_1['date'])
type(data_1['date'].iloc[0])
data_1['team1'].unique()
data_1.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
data_1['team1'].unique()
print('seasons:', data_1.season.unique())
print('Cities:', data_1.city.unique())
print(data_1.result.unique())
#Number of Tie matches



print("No. of Tie matches: "+str(data_1[data_1.result=='tie'].id.count()))
#No result matches



print("No. of no result matches: "+str(data_1[data_1.result=='no result'].id.count()))
#matches per season

data_1.groupby('season')['season'].count()
sns.countplot(x='season', data=data_1)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='venue', data=data_1)

plt.xticks(rotation='vertical')

plt.show()
temp = pd.melt(data_1, id_vars=['id','season'], value_vars=['team1', 'team2'])



plt.figure(figsize=(12,6))

sns.countplot(x='value', data=temp)

plt.xticks(rotation='vertical')

plt.show()
plt.subplots(figsize=(15,9))

toss = data_1.toss_decision.value_counts()

labels = (np.array(toss.index))

sizes = (np.array((toss / toss.sum())*100))

colors = ['lightgreen', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

plt.title("Toss decision percentage")

plt.show()
#no of matches where toss winner is the match winner

print("No of matches where toss winner is match winner: "+str(data_1[(data_1.result == 'normal') & (data_1.toss_winner == data_1.winner)].id.count()))

print("No of matches where toss winner is not match winner: "+str(data_1[(data_1.result == 'normal') & (data_1.toss_winner != data_1.winner)].id.count()))
plt.subplots(figsize=(15,9))

data_1['toss_winner_is_winner'] = 'no'

data_1['toss_winner_is_winner'].loc[data_1.toss_winner == data_1.winner] = 'yes'

result = data_1.toss_winner_is_winner.value_counts()



labels = (np.array(result.index))

sizes = (np.array((result / result.sum())*100))

colors = ['lightgreen', 'lightskyblue']

plt.pie(sizes, labels=labels, colors=colors,

        autopct='%1.2f%%', shadow=True, startangle=90)

plt.title("Toss winner is match winner")

plt.show()
plt.subplots(figsize=(12,8))

ax=data_1['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('Blues_d',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='toss_decision', data=data_1)

plt.xticks(rotation='vertical')

plt.show()
data_1["field_win"] = "win"

data_1["field_win"].loc[data_1['win_by_wickets']==0] = "loss"

plt.figure(figsize=(12,6))

sns.countplot(x='season', hue='field_win', data=data_1)

plt.xticks(rotation='vertical')

plt.show()
data_1.groupby("winner")['winner'].count().plot(figsize=(12,12),kind='pie',autopct='%1.1f%%',shadow=False)

#data_1.groupby("winner")['winner'].count().plot(figsize=(12,12),kind='pie',autopct='%1.1f%%',shadow=True)
data_1.groupby('winner')['winner'].agg(['count']).sort_values('count').reset_index().plot(x='winner',y='count',kind='barh')
plt.figure(figsize=(12,6))

sns.countplot(x='winner', data=data_1)

plt.xticks(rotation='vertical')

plt.show()
df1 = pd.DataFrame({"count":data_1.groupby('player_of_match')['player_of_match'].count()}).reset_index()

df1
#sorting the values

df1 = df1.sort_values('count',ascending=False)
df1[0:10].plot.barh(figsize=(20,10),x='player_of_match',y='count',xticks=[2,4,6,8,10,12,14,16,18,20])
matches_played_KKR=data_1[(data_1['team1']=='Kolkata Knight Riders') | (data_1['team2']=='Kolkata Knight Riders')]

matches_played_MI=data_1[(data_1['team1']=='Mumbai Indians') | (data_1['team2']=='Mumbai Indians')]

matches_played_KXP=data_1[(data_1['team1']=='Kings XI Punjab') | (data_1['team2']=='Kings XI Punjab')]

matches_played_CSK=data_1[(data_1['team1']=='Chennai Super Kings') | (data_1['team2']=='Chennai Super Kings')]

matches_played_DC=data_1[(data_1['team1']=='Deccan Chargers') | (data_1['team2']=='Deccan Chargers')]

matches_played_DD=data_1[(data_1['team1']=='Delhi Daredevils') | (data_1['team2']=='Delhi Daredevils')]

matches_played_RCB=data_1[(data_1['team1']=='Royal Challengers Bangalore') | (data_1['team2']=='Royal Challengers Bangalore')]

matches_played_KT=data_1[(data_1['team1']=='Kochi Tuskers Kerala') | (data_1['team2']=='Kochi Tuskers Kerala')]

matches_played_SH=data_1[(data_1['team1']=='Sunrisers Hyderabad') | (data_1['team2']=='Sunrisers Hyderabad')]

matches_played_RPS=data_1[(data_1['team1']=='Rising Pune Supergiants') | (data_1['team2']=='Rising Pune Supergiants')]

matches_played_RR=data_1[(data_1['team1']=='Rajasthan Royals') | (data_1['team2']=='Rajasthan Royals')]



A=matches_played_KKR['id'].count()

B=matches_played_MI['id'].count()

C=matches_played_KXP['id'].count()

D=matches_played_CSK['id'].count()

E=matches_played_DC['id'].count()

F=matches_played_DD['id'].count()

G=matches_played_RCB['id'].count()

H=matches_played_KT['id'].count()

I=matches_played_SH['id'].count()

J=matches_played_RPS['id'].count()

K=matches_played_RR['id'].count()

matches_won_KKR=data_1[data_1['winner']=='Kolkata Knight Riders']

matches_won_MI=data_1[data_1['winner']=='Mumbai Indians']

matches_won_KXP=data_1[data_1['winner']=='Kings XI Punjab']

matches_won_CSK=data_1[data_1['winner']=='Chennai Super Kings']

matches_won_DC=data_1[data_1['winner']=='Deccan Chargers']

matches_won_DD=data_1[data_1['winner']=='Delhi Daredevils']

matches_won_RCB=data_1[data_1['winner']=='Royal Challengers Bangalore']

matches_won_KT=data_1[data_1['winner']=='Kochi Tuskers Kerala']

matches_won_SH=data_1[data_1['winner']=='Sunrisers Hyderabad']

matches_won_RPS=data_1[data_1['winner']=='Rising Pune Supergiants']

matches_won_RR=data_1[data_1['winner']=='Rajasthan Royals']





O=matches_won_KKR['id'].count()

P=matches_won_MI['id'].count()

Q=matches_won_KXP['id'].count()

R=matches_won_CSK['id'].count()

S=matches_won_DC['id'].count()

T=matches_won_DD['id'].count()

U=matches_won_RCB['id'].count()

V=matches_won_KT['id'].count()

W=matches_won_SH['id'].count()

X=matches_won_RPS['id'].count()

Y=matches_won_RR['id'].count()



n_bins = 11

ind = np.arange(n_bins)

width = 0.50





matches_played=[A,B,C,D,E,F,G,H,I,J,K]

matches_won=[O,P,Q,R,S,T,U,V,W,X,Y]



#matches_played.sort()

#matches_won.sort()

plt.figure(figsize=(10,10))



p1 = plt.bar(ind, matches_played, width, color='LightSkyBlue')

p2 = plt.bar(ind, matches_won, width, color='lightgreen')



plt.ylabel('Number of Matches')

plt.xlabel('IPL teams')

plt.title('Overall performance of the team')

plt.xticks(ind + width/2., ('KKR', 'MI', 'KXP', 'CSK', 'DC', 'DD', 'RCB', 'KT', 'SH', 'RPS', 'RR'))

plt.yticks(np.arange(0, 200, 5))

plt.legend((p1[0], p2[0]), ('matches_played', 'matches_won'))
ump = pd.melt(data_1, id_vars=['id'], value_vars=['umpire1', 'umpire2'])



ump = ump.value.value_counts()[:10]

labels = np.array(ump.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(ump), width=width, color='lightblue')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top Umpires")

plt.show()
data_2.head()
data_2.shape
data_2.isnull().sum()
df_bats = data_2[data_2.batsman == 'MS Dhoni']

df_bats.head()
df_bats.groupby('match_id')['batsman_runs'].sum().plot(figsize=(20,10),kind = 'line')

data_2[data_2.player_dismissed == 'MS Dhoni'].groupby('dismissal_kind')['dismissal_kind'].count().plot(kind='bar')
data_2[data_2.player_dismissed == 'MS Dhoni'].groupby('bowler')['bowler'].count().plot(figsize=(20,10),kind='bar')
df1 = pd.DataFrame({'wickets':data_2[data_2.player_dismissed.notna()].groupby('bowler')['bowler'].count()}).reset_index()

df1= df1.sort_values('wickets',ascending=False)

print(df1[0:10])

df1[0:10].groupby('bowler')['wickets'].sum().plot(kind='barh',figsize=(20,10))

data_2[data_2.bowler == 'SL Malinga'].groupby('dismissal_kind')['dismissal_kind'].count().plot(kind='bar')


maximum_runs = data_2.groupby(['batsman'])['batsman_runs'].sum()

maximum_runs

maximum_runs.sort_values(ascending = False,inplace=True)

plt.figure(figsize=(8,10))

maximum_runs[:10].plot(x= 'bowler', y = 'runs', kind = 'bar', colormap = 'Pastel2')

plt.xlabel('Batsmen')

plt.ylabel('Most Runs in IPL')
plt.figure(figsize=(13,7))

ax=sns.countplot(data_2.dismissal_kind)

plt.xticks(rotation=90)

overs, number = np.unique(np.concatenate(data_2.groupby(['match_id','inning'])['over'].unique().values), return_counts=True)

average_runs_in_each_over = ((data_2.groupby(['over'])['total_runs'].sum())/(number)).round(2)
plt.figure(figsize=(15,5))

sns.set_style("whitegrid")

ax = sns.barplot(x=average_runs_in_each_over.index,y=average_runs_in_each_over.values,palette='Blues')

ax.set_xlabel("Overs").set_size(20)

ax.set_ylabel("Runs").set_size(20)

ax.set_title("Average runs in each over").set_size(20)