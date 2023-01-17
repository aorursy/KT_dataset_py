# Importing Essential Libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='ignore')
df = pd.read_csv('/kaggle/input/ipl/matches.csv')

df.head()
df.city.fillna('Dubai',inplace=True)
df.winner.fillna('Match Abandoned',inplace=True)

df.player_of_match.fillna('Match Abandoned',inplace=True)

df.umpire1.fillna('Anonymous',inplace=True)

df.umpire2.fillna('Anonymous',inplace=True)

df.drop(columns='umpire3',inplace=True)
df['winner'][df['winner'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'

df['team1'][df['team1'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'

df['team2'][df['team2'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'

df['toss_winner'][df['toss_winner'] == 'Rising Pune Supergiants'] = 'Rising Pune Supergiant'
teams = pd.Series(df['team1'].unique())
years = pd.Series(df['season'].unique())

years=years.sort_values()
total_matches_played_dict = {}



for team in teams:

    matches_per_year = []

    for year in years:

        total_match_per_year = df[((df['team1']==team) | (df['team2']==team)) & (df['season']==year)].shape[0]

        matches_per_year.append(total_match_per_year)

    total_matches_played_dict[team] = matches_per_year

dframe_matches_played=pd.DataFrame(total_matches_played_dict,index=years)
dframe_matches_played
dict_win_num = {}



for team in teams:

    matches_won = []

    for year in years:

        matches_won_per_year = df[(df['winner']==team) & (df['season']==year)].shape[0]

        matches_won.append(matches_won_per_year)

    dict_win_num[team] = matches_won

dframe_matches_won = pd.DataFrame(dict_win_num,index=years)
dframe_matches_won
dframe_match_won_percnt = (dframe_matches_won/dframe_matches_played)*100

dframe_match_won_percnt.fillna(0,inplace=True)
dframe_match_won_percnt
dframe_match_won_percnt['Year'] = dframe_match_won_percnt.index
dict_oppnt_matches = {}



for team_1 in teams:

    matches_against_oppnt = []

    for team_2 in teams:

        match_plyd_against_this_oppnt = df[((df['team1']==team_1) & (df['team2']==team_2)) | ((df['team1']==team_2) & (df['team2']==team_1))].shape[0]

        matches_against_oppnt.append(match_plyd_against_this_oppnt)

    dict_oppnt_matches[team_1] = matches_against_oppnt

    

dframe_total_match_agnst_oppnt = pd.DataFrame(dict_oppnt_matches,index=teams)
dframe_total_match_agnst_oppnt
dict_oppnt_matches_won = {}



for team_1 in teams:

    matches_won_against_oppnt = []

    for team_2 in teams:

        match_won_against_this_oppnt = df[(((df['team1']==team_1) & (df['team2']==team_2)) | ((df['team1']==team_2) & (df['team2']==team_1))) & (df['winner']==team_1)].shape[0]

        matches_won_against_oppnt.append(match_won_against_this_oppnt)

    dict_oppnt_matches_won[team_1] = matches_won_against_oppnt

    

dframe_total_match_won_agnst_oppnt = pd.DataFrame(dict_oppnt_matches_won,index=teams)
dframe_total_match_won_agnst_oppnt
dframe_oppnt_won_percnt = (dframe_total_match_won_agnst_oppnt/dframe_total_match_agnst_oppnt)*100

dframe_oppnt_won_percnt.fillna(0,inplace=True)
dframe_oppnt_won_percnt
teams_dict = {'Sunrisers Hyderabad' : 'SRH',

'Mumbai Indians' : 'MI',

'Gujarat Lions' : 'GL',

'Rising Pune Supergiant' : 'RPS',

'Royal Challengers Bangalore' : 'RCB',

'Kolkata Knight Riders' : 'KKR',

'Delhi Daredevils' : 'DD',

'Kings XI Punjab' : 'KXIP',

'Chennai Super Kings' : 'CSK',

'Rajasthan Royals' : 'RR',

'Deccan Chargers' : 'DC',

'Kochi Tuskers Kerala' : 'KTK',

'Pune Warriors' : 'PW'}
dframe_oppnt_won_percnt['Teams'] = dframe_oppnt_won_percnt.index.map(teams_dict)
dframe_oppnt_won_percnt
sns.set_style('whitegrid')
plt.figure(figsize=(18,6))

plt.title('Deccan Chargers Graph over the Years')

plt.ylim(0,100)

plt.xlim(2008,2012)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Deccan Chargers',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Deccan Chargers performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Deccan Chargers')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Deccan Chargers against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Deccan Chargers')
plt.figure(figsize=(16,6))

plt.title('Mumbai Indians Graph over the years')

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

sns.lineplot(

    x='Year',

    y='Mumbai Indians',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Mumbai Indians performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Mumbai Indians')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Mumbai Indians against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Mumbai Indians')
plt.figure(figsize=(16,6))

plt.title('Royal Challengers Bangalore Graph over the Years')

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Royal Challengers Bangalore',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Royal Challengers Bangalore performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Royal Challengers Bangalore')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Royal Challengers Bangalore against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Royal Challengers Bangalore')
plt.figure(figsize=(18,6))

plt.title('Chennai Super Kings Graph over the Years')

plt.ylim(0,100)

plt.xlim(2008,2015)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Chennai Super Kings',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

plt.axhline(y=50,color='red')

plt.title('Chennai Super Kings performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Chennai Super Kings')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Chennai Super Kings against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Chennai Super Kings')
plt.figure(figsize=(18,6))

plt.title('Kings XI Punjab Graph over the Years')

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Kings XI Punjab',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

plt.axhline(y=50,color='red')

plt.title('Kings XI Punjab performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Kings XI Punjab')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Kings XI Punjab against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Kings XI Punjab')
plt.figure(figsize=(18,6))

plt.title('Delhi Daredevils Graph over the Years')

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Delhi Daredevils',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

plt.axhline(y=50,color='red')

plt.title('Delhi Daredevils performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Delhi Daredevils')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Delhi Daredevils against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Delhi Daredevils')
plt.figure(figsize=(18,6))

plt.title('Kolkata Knight Riders Graph over the Years')

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Kolkata Knight Riders',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

plt.axhline(y=50,color='red')

plt.title('Kolkata Knight Riders performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Kolkata Knight Riders')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Kolkata Knight Riders against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Kolkata Knight Riders')
plt.figure(figsize=(18,6))

plt.title('Sunrisers Hyderabad Graph over the Years')

plt.ylim(0,100)

plt.xlim(2013,2017)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Sunrisers Hyderabad',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

plt.axhline(y=50,color='red')

plt.title('Sunrisers Hyderabad performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Sunrisers Hyderabad')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Sunrisers Hyderabad against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Sunrisers Hyderabad')
plt.figure(figsize=(18,6))

plt.title('Rajasthan Royals Graph over the Years')

plt.ylim(0,100)

plt.xlim(2008,2015)

plt.axhline(y=50,color='red',linewidth='0.6')



sns.lineplot(

    x='Year',

    y='Rajasthan Royals',

    data=dframe_match_won_percnt)
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red',linewidth='0.6')

plt.axhline(y=50,color='red')

plt.title('Rajasthan Royals performance over the years')



sns.barplot(data=dframe_match_won_percnt,x='Year',y='Rajasthan Royals')
plt.figure(figsize=(16,6))

plt.ylim(0,100)

plt.axhline(y=50,color='red')

plt.title('Rajasthan Royals against its Opponents')



sns.barplot(data=dframe_oppnt_won_percnt,x='Teams',y='Rajasthan Royals')