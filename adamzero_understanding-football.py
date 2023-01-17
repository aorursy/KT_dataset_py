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



event_type_dic = {0 :'Announcement', 1 :'Attempt', 2 : 'Corner', 3: 'Foul', 4 : 'Yellow card', 5 : 'Second yellow card',

6 : 'Red card', 7 : 'Substitution', 8 : 'Free kick won', 9 : 'Offside', 10 : 'Hand ball', 11 : 'Penalty conceded'}

event_type2_dic = {12 : 'Key Pass', 13 : 'Failed through ball', 14 : 'Sending off', 15 : 'Own goal'}

side_dic = {1 : 'Home', 2 : 'Away'}

shot_place_dic ={ 1 : 'Bit too high', 2 : 'Blocked', 3 : 'Bottom left corner', 4 : 'Bottom right corner', 5 : 'Centre of the goal',

6 : 'High and wide', 7 : 'Hits the bar', 8 : 'Misses to the left', 9 : 'Misses to the right', 10 : 'Too high', 11 : 'Top centre of the goal',

12 : 'Top left corner', 13 : 'Top right corner'}

shot_outcome_dic ={1 : 'On target', 2 :'Off target', 3 : 'Blocked', 4 : 'Hit the bar'}

location_dic = {1 : 'Attacking half', 2 : 'Defensive half', 3 : 'Centre of the box', 4 :'Left wing', 5 : 'Right wing',

                6 : 'Difficult angle and long range', 7 : 'Difficult angle on the left', 8 : 'Difficult angle on the right',

                9 : 'Left side of the box', 10 : 'Left side of the six yard box', 11 : 'Right side of the box',

                12 : 'Right side of the six yard box', 13 : 'Very close range', 14 : 'Penalty spot', 15 : 'Outside the box',

                16 : 'Long range', 17 : 'More than 35 yards', 18 : 'More than 40 yards', 19 : 'Not recorded'}

bodypart_dic = {1 :'right foot', 2 : 'left foot', 3 : 'head'}

assist_method_dic = {0 : 'None',1 : 'Pass',2 : 'Cross',3 : 'Headed pass',4 : 'Through ball'}



situation_dic = {1 : 'Open play', 2 : 'Set piece', 3 : 'Corner', 4 : 'Free kick'}



event_df = pd.read_csv('../input/football-events/events.csv')

ginf_df = pd.read_csv('../input/football-events/ginf.csv')
ginf_df.info()
event_df.info()
event_df.replace({"event_type": event_type_dic}, inplace=True)

event_df.replace({"event_type2": event_type2_dic}, inplace=True)

event_df.replace({"side": side_dic}, inplace=True)

event_df.replace({"shot_place": shot_place_dic}, inplace=True)

event_df.replace({"shot_outcome": shot_outcome_dic}, inplace=True)

event_df.replace({"location": location_dic}, inplace=True)

event_df.replace({"bodypart": bodypart_dic}, inplace=True)

event_df.replace({"assist_method": assist_method_dic}, inplace=True)

event_df.replace({"situation": situation_dic}, inplace=True)
ginf_df.rename(columns={'ht': 'Home Team', 'at': 'Away Team', 'fthg' : 'Home Team Score', 'ftag' : 'Away Team Score',

                       'odd_h' : 'Home win odds', 'odd_d' : 'Draw odds', 'odd_a' : 'Away win odds'}, inplace=True)

ginf_df.columns
ginf_df['league'].unique()
sns.distplot(event_df['time']);
ginf_df.drop(['link_odsp', 'adv_stats', 'odd_over', 'odd_under', 'odd_bts', 'odd_bts_n'], axis=1, inplace=True)

event_df.drop(['id_event', 'sort_order', 'text'], axis=1, inplace=True)

#event_df.drop(['player', 'player2'], axis=1, inplace=True)

event_df.drop(['player_in', 'player_out'], axis=1, inplace=True)
df = pd.merge(left=ginf_df, right=event_df, on="id_odsp", how="left")

shot_df = event_df[event_df['shot_place'].notnull()]
sns.distplot(shot_df['time']);
shot_df['is_goal'].mean()
shot_df.groupby("bodypart")["is_goal"].mean()
shot_df.groupby("side")["is_goal"].mean()
shot_df.groupby("shot_place")["is_goal"].mean()
shot_df.groupby("shot_place")["is_goal"].sum()
plt.figure(figsize=(15,10))

sns.countplot(y = shot_df["side"],hue=shot_df["shot_place"],

              palette=["r","g","b","c","lime","m","y","k","gold","orange"])

plt.title("Shot Placement - Home and Away")

plt.show()
shot_df.groupby("assist_method")["is_goal"].mean()
df.columns
df[['id_odsp', 'Home Team', 'Away Team', 'time', 'event_type', 'Home Team Score',

       'Away Team Score', 'Home win odds', 'Draw odds', 'Away win odds' ]]
match_summ = ginf_df.copy()

match_summ.drop(['date', 'league'], axis=1, inplace=True)

home_summ =df[df['Home Team']== df['event_team']].assign(cts=1).pivot_table(index=['id_odsp', 'Home Team', 'Away Team', 'Home Team Score',

       'Away Team Score', 'Home win odds', 'Draw odds', 'Away win odds' ], columns='event_type', values='cts', aggfunc='sum').fillna(0)

away_summ = df[df['Home Team'] != df['event_team']].assign(cts=1).pivot_table(index=['id_odsp'], columns='event_type', values='cts', aggfunc='sum').fillna(0)



#home_summ.rename(columns = {'Attempt':'Home Attempt', 'Corner' : 'Home Corner', 'Foul' : 'Home Foul', 'Free kick won' : 'Home Free kick won',

#                            'Hand ball' : 'Home Hand ball', 'Offside' : 'Home Offside', 'Penalty conceded' : 'Home Penalty conceded', 'Red card' : 'Home Red card',

#                            'Second yellow card' : 'Home Second yellow card', 'Substitution' : 'Home Substitution','Yellow card' : 'Home Yellow card'}, inplace = True)

#away_summ.rename(columns = {'Attempt':'Away Attempt', 'Corner' : 'Away Corner', 'Foul' : 'Away Foul', 'Free kick won' : 'Away Free kick won',

#                            'Hand ball' : 'Away Hand ball', 'Offside' : 'Away Offside', 'Penalty conceded' : 'Away Penalty conceded', 'Red card' : 'Away Red card',

#                            'Second yellow card' : 'Away Second yellow card', 'Substitution' : 'Away Substitution','Yellow card' : 'Away Yellow card'}, inplace = True)



match_df = pd.DataFrame()

match_df = pd.merge(left=home_summ, right=away_summ, on="id_odsp", how="left")

match_df = pd.merge(left=ginf_df[['id_odsp', 'Home Team', 'Away Team','Home Team Score','Away Team Score', 'country','season', 'Home win odds', 'Draw odds', 'Away win odds' ]], right=match_df, on="id_odsp", how="left")



match_df
match_df.groupby("country").agg({"Home Team Score":"mean", 'Away Team Score' : 'mean'}).plot(kind="barh",

                                                                                 figsize = (10,10),

                                                                                 edgecolor = "k",

                                                                                 linewidth =1

                                                                                )

plt.title("Home and away goals by league")

plt.legend(loc = "best" , prop = {"size" : 14})

plt.xlabel("total goals")

plt.show()
match_df.columns
plt.figure(figsize=(12,6))

sns.kdeplot(match_df["Foul_x"],shade=True,

            color="b",label="home fouls")

sns.kdeplot(match_df["Foul_y"],shade=True,

            color="r",label="away fouls")

plt.axvline(match_df["Foul_x"].mean(),linestyle = "dashed",

            color="b",label="home fouls mean")

plt.axvline(match_df["Foul_y"].mean(),linestyle = "dashed",

            color="r",label="away fouls mean")

plt.legend(loc="best",prop = {"size" : 12})

plt.title("Distribution of Home and Away Fouls")

plt.xlabel("Fouls")

plt.show()
plt.figure(figsize=(13,10))

plt.subplot(211)

sns.boxplot(x = match_df["season"],y = match_df["Attempt_y"],palette="rainbow")

plt.ylabel('Number of Home Attempts on Goal')

plt.title("Home Attempts by Season")

plt.subplot(212)

sns.boxplot(x = match_df["season"],y = match_df["Attempt_x"],palette="rainbow")

plt.ylabel('Number of Away Attempts on Goal')

plt.title("Away Attempts by Season")

plt.show()
Leicester_df = df[(df['event_team'] == 'Leicester City') | (df['opponent'] == 'Leicester City')] 

Leicester_event = df[df['event_team'] == 'Leicester City'] 

Leicester_opp = df[df['opponent'] == 'Leicester City'] 

Leicester_df
Leicester_event[Leicester_event['shot_place'].notnull()].groupby(['season']).mean()
Leicester_event[Leicester_event['shot_place'].notnull()].groupby(['season']).sum()
Leicester_event.groupby(['season']).sum()
plt.figure(figsize=(10,10))

sns.countplot(y = shot_df["side"],hue=shot_df["shot_place"],

              palette=["r","g","b","c","lime","m","y","k","gold","orange"])

plt.title("Shot Placement")

plt.show()
data = Leicester_event[Leicester_event['situation'].notnull()]

plt.figure(figsize=(10,10))

sns.countplot(y = data["season"],hue=data["situation"],

              palette=["r","g","b","c","lime","m","y","k","gold","orange"])

plt.title("shot place")

plt.show()
foxes_home_2015 = match_df[(match_df['season']== 2015) & (match_df['country']=='england') & (match_df['Home Team'] == 'Leicester City')]

foxes_home_2016 = match_df[(match_df['season']== 2016) & (match_df['country']=='england') & (match_df['Home Team'] == 'Leicester City')]



foxes_away_2015 = match_df[(match_df['season']== 2015) & (match_df['country']=='england') & (match_df['Away Team'] == 'Leicester City')]

foxes_away_2016 = match_df[(match_df['season']== 2016) & (match_df['country']=='england') & (match_df['Away Team'] == 'Leicester City')]
foxes_away_2015.mean()

pd.DataFrame({ 'Fox 2015 home ':foxes_home_2015.mean(),'Fox 2016 home ':foxes_home_2016.mean(),

              'Fox 2015 away ':foxes_away_2015.mean(),'Fox 2016 away ':foxes_away_2016.mean(), 'General':match_df.mean()})
Vardy_df = df[(df['event_team'] == 'Leicester City') | (df['opponent'] == 'Leicester City')] 

Vardy_df =Vardy_df[Vardy_df['player']== 'jamie vardy']

Vardy_df.head()
#Vardy_df.drop(['link_odsp', 'adv_stats', 'odd_over', 'odd_under', 'odd_bts', 'odd_bts_n'], axis=1, inplace=True)

#Vardy_df.drop(['id_event', 'sort_order'], axis=1, inplace=True)

#Vardy_df.drop(['player_in', 'player_out'], axis=1, inplace=True)

Vardy_df.drop(['league', 'country'], axis=1, inplace=True)

Vardy_df.drop(['Home win odds', 'Draw odds', 'Away win odds'], axis=1, inplace=True)
Vardy_df.info()
Vardy_df.head()
#Vardy_df['event_type','season'].groupby('season').value_counts()

Vardy_df.groupby("season")["event_type"].value_counts()

plt.figure(figsize=(10,10))

sns.countplot(y = Vardy_df["season"],hue=Vardy_df["event_type"],

              palette=["r","g","b","c","lime","m","y","k","gold","orange"])

plt.title("Vardy Event Types")

plt.show()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby('season')["location"].value_counts()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby('season')["bodypart"].value_counts()
#Vardy_df[Vardy_df['is_goal']==1].groupby('season')["bodypart"].value_counts()

Vardy_df[Vardy_df['shot_place'].notnull()].groupby(['season', "bodypart"])['is_goal'].mean()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby('season')["assist_method"].value_counts()

#Vardy_df[Vardy_df['is_goal']==1].groupby('season')["assist_method"].value_counts()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby(['season',"assist_method"])['is_goal'].value_counts()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby('season')["situation"].value_counts()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby('season')["player2"].value_counts()
Vardy_df[Vardy_df['shot_place'].notnull()].groupby(['season', 'player2'])['is_goal'].mean()
Vardy_df[Vardy_df['is_goal']==1].groupby(['season'])['player2'].value_counts()
#Vardy_df.drop(['link_odsp', 'adv_stats', 'odd_over', 'odd_under', 'odd_bts', 'odd_bts_n','id_event', 'sort_order'], axis=1, inplace=True)
Vardy_df['assist_method'].unique()
Vardy_df.groupby(['season', 'assist_method'])['is_goal'].mean()
Vardy_df[(Vardy_df['is_goal']== 1.0) & (Vardy_df['season']==2015)]['time']