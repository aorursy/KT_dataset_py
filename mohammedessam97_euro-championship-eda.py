import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly.graph_objects as go

import plotly.express as px

%matplotlib inline 
matches = pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup All Matches.csv')

players = pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup All Players.csv')

teams_stats = pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup Participated Teams General Statistics.csv')

cups_stats=pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup General Statistics.csv')
matches.head()
matches.info()
players.head()
players.info()
teams_stats.head()
teams_stats.info()
cups_stats.head()
cups_stats.info()
matches['HomeTeamName'] = matches['HomeTeamName'].apply(lambda x : x.replace(u'\xa0', u'')).apply(lambda x : x.strip())

matches['AwayTeamName'] = matches['AwayTeamName'].apply(lambda x : x.replace(u'\xa0', u'')).apply(lambda x : x.strip())
matches.replace('Soviet Union','Russia',inplace=True)

matches.replace('West Germany','Germany',inplace=True)

cups_stats.replace('Soviet Union','Russia',inplace=True)

cups_stats.replace('West Germany','Germany',inplace=True)
games_by_city=matches.groupby(['City']).size()

games_by_city = games_by_city.reset_index()
games_by_city.head()

games_by_city.columns = ['City','Number of games']

games_by_city
top_cities = games_by_city.nlargest(10, ['Number of games']) 
plt.figure(figsize=(20,10))

fig = px.bar(top_cities, x='City', y='Number of games',color='Number of games')

fig.update_layout(title='Number of games played in each city in all tournaments',

                   xaxis_title='City',

                   yaxis_title='Number of Games')

fig.show()
top_attendance=matches[['City','Attendance','Year']].groupby(['Year']).max()
top_attendance=top_attendance.reset_index()
plt.figure(figsize=(20,10))

fig = go.Figure(data=[go.Bar(

            x=top_attendance['Year'], y=top_attendance['Attendance'],

            text=top_attendance['City'],

            textposition='outside',

        )])

fig.update_layout(title='Most Attendance in the tournament',

                   xaxis_title='Year',

                   yaxis_title='Attendance')

fig.show()
time_plot_1=go.Figure(go.Scatter(x=top_attendance['Year'], y=top_attendance['Attendance'],

                                 mode='lines+markers', line={'color': 'red'}))

time_plot_1.update_layout(title='Most Attendance in the tournament',

                   xaxis_title='Year',

                   yaxis_title='Attendance')

#showing the figure

time_plot_1.show()
matches.head()
matches['Year']=matches['Date'].apply(lambda x : x.split('(')[0]).apply(lambda x : x.split()[-1]).astype(int)
matches.head()
merged_data = pd.merge(matches,cups_stats, on=['Year', 'Year'])
merged_data.head()
merged_data = merged_data[['Year','HomeTeamName','AwayTeamName','Host','HomeTeamGoals','AwayTeamGoals']]
merged_data['Host'].tolist()
merged_data['Host']=merged_data['Host'].apply(lambda x : x.split())
splitted_data=pd.DataFrame([

    [year,Hometeam,awayteam, host, hgoals,agoals] for year,Hometeam,awayteam,Hosts, hgoals,agoals in merged_data.values

    for host in Hosts

], columns=merged_data.columns)
splitted_data[splitted_data['Year']==2012]
merged_data = splitted_data
merged_data = merged_data[(merged_data['HomeTeamName']==merged_data['Host'])|(merged_data['AwayTeamName']==merged_data['Host'])]

home_win = merged_data['HomeTeamGoals']>merged_data['AwayTeamGoals']

home_name = merged_data['HomeTeamName']==merged_data['Host']

away_win = merged_data['AwayTeamGoals']>merged_data['HomeTeamGoals']

away_name = merged_data['AwayTeamName']==merged_data['Host']
merged_data['Wins']=((home_win & home_name)|(away_win & away_name))
merged_data.head()
final_data = merged_data.groupby(['Year','Host']).sum()['Wins']/merged_data.groupby(['Year']).count()['Wins']
final_data = final_data.reset_index()
final_data['Wins']=final_data['Wins'] * 100 
plt.figure(figsize=(20,10))

fig = px.bar(final_data, x='Year', y='Wins',color='Wins',hover_data=['Host'])

fig.update_layout(title='Win Percentage for the Hosts',

                   xaxis_title='Year',

                   yaxis_title='Win Percentage')

fig.show()