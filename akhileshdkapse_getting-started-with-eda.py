# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/ipldata/matches.csv')
data.info()
data.describe()
data.head()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
plt.figure(figsize=(10,5))
data.city.value_counts().head(10).plot(kind='bar')
plt.ylabel('counts')
plt.title('Demanding cities')
#top 10 cities
data.player_of_match.value_counts().head()
plt.figure(figsize=(18,6))
for name,count in zip(data.player_of_match.value_counts().index[0:15],data.player_of_match.value_counts().head(15)):
    plt.text(name,count/2,str(name)+':'+str(count),rotation=90,color='white')
plt.bar(data.player_of_match.value_counts().index[0:15],data.player_of_match.value_counts().head(15),width=.5)
plt.title('top 15 man of the match winng players')
plt.ylabel('no. of awards')
win_score=data.groupby('season').win_by_runs.max()
plt.figure(figsize=(15,5))
plt.scatter(x=win_score.index.values,y=win_score.values)
plt.plot(win_score.index.values,win_score.values)
plt.xticks(win_score.index)
for ind,score in zip(win_score.index.values,win_score.values):
    plt.text(ind,score,str(score))
plt.xlabel('Season',fontsize=14)
plt.ylabel('Runs',fontsize=14)
plt.title('Biggest win by runs per season',fontsize=14)    
def win_count(team_nam):
    df1=pd.DataFrame([])
    tot=[]
    for key,dta in data.groupby(['season']):
        df=dta[(dta.team1==team_nam)|(dta.team2==team_nam)&(dta.winner==team_nam)]
        df1[key]=df.shape[0:1]
        tot.append(dta[(dta.team1==team_nam)|(dta.team2==team_nam)].shape[0])
    
    return df1 ,tot   
for team in data.team1.unique().tolist():
    
    df,total=win_count(team)
    plt.figure(figsize=(12,5))
    x=list(df.columns.values)
    plt.plot(x,total)
    plt.scatter(x,total)
    plt.title(team)
    plt.scatter(x,df.iloc[0].values)
    plt.xlabel('SEASON')
    plt.ylabel('WINNING count')
    plt.plot(x,df.iloc[0].values)
    
    plt.xticks(x)
df=data[['season','team1','team2','toss_winner','winner']]
df.head()
def one_vs_one_per_season(t_a,t_b):
    df2=df[((df.team1==t_a) & (df.team2==t_b)) | ((df.team1==t_b) & (df.team2==t_a))]
    ser=df2.groupby('season').winner.value_counts()
    return pd.DataFrame(ser)
one_vs_one_per_season('Mumbai Indians','Kolkata Knight Riders')
def one_vs_one_all_season(t_a,t_b):
    df2=df[((df.team1==t_a) & (df.team2==t_b)) | ((df.team1==t_b) & (df.team2==t_a))]
    plt.title('Match winning persentage over total matches')
    plt.pie(df2.winner.value_counts().values,shadow=True,labels=[t_a,t_b],explode=(.1,0),autopct='%.1f%%',)
    
one_vs_one_all_season('Mumbai Indians','Kolkata Knight Riders')

def one_vs_one_toss_win(t_a,t_b):
    df2=df[((df.team1==t_a) & (df.team2==t_b)) | ((df.team1==t_b) & (df.team2==t_a))]
    plt.title('Toss winning persentage over total matches')
    plt.pie(df2.toss_winner.value_counts().values,shadow=True,labels=[t_a,t_b],explode=(.1,0),autopct='%.1f%%',)
    
one_vs_one_toss_win('Mumbai Indians','Kolkata Knight Riders')
