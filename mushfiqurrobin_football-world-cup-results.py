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
df = pd.read_csv('../input/world_cup_results.csv')
df.head()
import pandas as pd
import matplotlib as plt
df.shape
#let's deal with the duplicates
#Notice that from a shape of (852cols, 11rows) we now arrive at (836, 11). There was duplicates
df = df.drop_duplicates()
df.shape
#Total goals column
df['TotalGoals'] = df['HomeGoals'] + df['AwayGoals']
df['month'] = df['Date'].apply(lambda x: x.split('-')[1])
df['day'] = pd.to_datetime(df['Date']).dt.day_name()
#Let's see what our df looks like now by peeping the head
df.head(2)
df = df[['Year', 'month', 'day', 'Time', 'Round', 'HomeTeam', 'HomeGoals', 'AwayTeam', 'AwayGoals', 'TotalGoals']]
df.head(2)
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#A value_count on the Year column nicely delivers this
#To plot this effortlessly with plotly, we will convert the result to a fresh dataframe
#Notice how nicely plotly highlights the expected years world cups were not played
matches_per_year = df.Year.value_counts() #a series to hold our values
all_games = pd.DataFrame(matches_per_year) #make series into a dataframe
all_games.reset_index(inplace=True) #reset it's index inplace
all_games.columns = ['Year', 'Matches'] #rename the columns as needed
fig = px.bar(all_games, x='Year', y='Matches', text='Matches', color='Matches', height=500,
labels={'Matches':'Matches Played', 'Year':'World Cup Year'},
title="Total Matches Played Each World Cup Year")
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout()
fig.update_xaxes(
tickangle=45, tickfont=dict(family='Arial', color='blue', size=14),
tickvals=[line for line in all_games.Year])
fig.show()
all_goals = df.groupby(['Year']).TotalGoals.sum()
all_goals_df = pd.DataFrame(all_goals)
all_goals_df.reset_index(inplace=True) #reset it's index inplace
all_goals_df.columns = ['Year', 'Goals'] #rename the columns as needed
fig = px.bar(all_goals_df, x='Year', y='Goals', text='Goals', color='Goals', height=450,
labels={'Goals':'Goals Scored', 'Year':'World Cup Year'},
title="Total Goals Scored Each World Cup Year")
fig.update_traces(texttemplate='%{text}', textposition='inside')
fig.update_layout()
fig.update_xaxes(
tickangle=45, tickfont=dict(family='Arial', color='blue', size=14),
tickvals=[line for line in all_goals_df.Year])
fig.show()
#Take a piece of the df corresponding to all 'Final' in the 'Round' column
all_finals = df[df['Round'] == 'Final']
all_finals.head()
#Let's make a list of all teams who reach this stage.
#This will be a list of all featuring HomeTeams and AwayTeams
#A simple concatenation of a python list of both will do
teams = [line for line in all_finals.HomeTeam] + [line for line in all_finals.AwayTeam]
#Peep a sample
teams[:5]
#To count the frequency that I am going to eventually plot I prefer to use a dataframe. It's seamless
#So I will make a dataframe from the list 'teams' and take a drop a value_counts(). Neat, yea?
all_finals_df = pd.DataFrame(columns=['Teams'], data = teams)
#peep the head()
all_finals_df.head(2)
#To demonstrate this value_counts() counting, see the result before we plot
#PS: I am choosing to leave Germany and Germany FR as different entities
all_finals_df.Teams.value_counts()
#Make a dataframe from counting values in all_finals_df
finals_teams_ranked = all_finals_df.Teams.value_counts()
finals_teams_ranked_df = pd.DataFrame(finals_teams_ranked)
finals_teams_ranked_df.reset_index(inplace=True)
finals_teams_ranked_df.columns = ['Teams', 'Frequency'] #rename the columns as needed
fig = px.bar(finals_teams_ranked_df, x='Teams', y='Frequency', color='Frequency', height=450,
labels={'Teams':'Teams in the Finals'},
title="All Teams Who Have Reached Finals and Frequency")
fig.update_layout()
fig.show()
#Take a piece of the df corresponding to all 'Semi-finals' in the 'Round' column
all_semi_finals = df[df['Round'] == 'Semi-finals']
all_semi_finals.head()
#Make a list of all teams invovled Home and Away
teams = [line for line in all_semi_finals.HomeTeam] + [line for line in all_semi_finals.AwayTeam]
#Make a df of teams
all_semi_finals_df = pd.DataFrame(columns=['Teams'], data = teams)
#peep the head()
all_semi_finals_df.head(2)
#Make a dataframe from counting values in all_semi_finals
#See 'Finals' cell above for explanation as the steps are identical. We are avoiding functions for practise
semifinals_teams_ranked = all_semi_finals_df.Teams.value_counts()
semifinals_teams_ranked_df = pd.DataFrame(semifinals_teams_ranked)
semifinals_teams_ranked_df.reset_index(inplace=True)
semifinals_teams_ranked_df.columns = ['Teams', 'Frequency']
fig = px.bar(finals_teams_ranked_df, x='Teams', y='Frequency', color='Frequency', height=450,
labels={'Teams':'Teams in the Finals'},
title="All Teams Who Have Reached Semi-Final and Frequency")
fig.update_layout()
fig.show()
#Continue with our dataframe holding only semi-final matches
#Let's take a sum and mean of 'TotalGoals'
semi_goals_sum = all_semi_finals.TotalGoals.sum()
semi_goals_ave = all_semi_finals.TotalGoals.mean()
print(f"{semi_goals_sum} goals were scored in all Semi-Finals\nAn average of {semi_goals_ave:.2f} in every match.")
#Get a slice of the original df for all quarter-finals
all_qtrs = df[df['Round'] == 'Quarter-finals']
all_qtrs.head()
qtrs_goals_sum = all_qtrs.TotalGoals.sum()
qtrs_goals_ave = all_qtrs.TotalGoals.mean()
print(f"{qtrs_goals_sum} goals were scored in all Semi-Finals\nAn average of {qtrs_goals_ave:.2f} in every match.")
#All goals in finals
finals = df[df['Round'] == 'Final']['HomeGoals'].sum() + df[df['Round'] == 'Final']['AwayGoals'].sum()#[['HomeTeam',
print(f"{finals} goals in finals")
#Working with the piece of the original df holding finals
finals_goals_sum = all_finals.TotalGoals.sum()
finals_goals_ave = all_finals.TotalGoals.mean()
print(f"{finals_goals_sum} goals were scored in all Semi-Finals\nAn average of {finals_goals_ave:.2f} in every match")
#Let's be creative here!
#First get a slice with no finals
df_less_finals = df[df['Round'] != 'Final']
#From there get a slies with no semi-finals and viola we are left with all matches neither finals or semis
df_less_finals_semis = df_less_finals[df_less_finals['Round'] != 'Semi-Finals']
#One more dropping qtrs. This is fun.
df_less_finals_semis_qtrs = df_less_finals_semis[df_less_finals_semis['Round'] != 'Quarter-finals']
#Did it work? Well, let's check!
'Final' in df_less_finals_semis_qtrs.Round.tolist() or 'Semi-Finals' in df_less_finals_semis_qtrs.Round.tolist()
#Just in case that was lady-luck, let's make sure other Rounds are there
'Round of 16' in df_less_finals_semis_qtrs.Round.tolist()
#Total matches in this slice of the dataframe is same number of rows. A number of move will show the number
d_rest0 = df_less_finals_semis_qtrs.shape[0]
d_rest1 = len(df_less_finals_semis_qtrs)
d_rest0 == d_rest1
print(f"There are {d_rest0} matches played outside Quater-finals and above")
#I love python lists a lot as I know them in and out
#I will use a zip of four different columns from the datafarame to solve the kicker
# a python list of all four columns we are considering
AwayT_list = df['AwayTeam'].tolist()
HomeT_list = df['HomeTeam'].tolist()
AwayG_list = df['AwayGoals'].tolist()
HomeG_list = df['HomeGoals'].tolist()
#Two empty lists to hold our values for the two new columns
verdict, winner = [], []
#We zip the four lists created and step through them looking for the kicker condition, assigning values as we go
for at, ht, ag, hg in zip (AwayT_list, HomeT_list, AwayG_list, HomeG_list):
    if ag > hg:
       verdict.append('A')
       winner.append (at)
    elif hg > ag:
       verdict.append('H')
       winner.append(ht)
    elif hg == ag:
     if ag == 0:
      verdict.append('D')
      winner.append('Draw')
     else:
      verdict.append('A')
      winner.append(at)
#Finally write the two new columns to our dataframe
df['Verdict'] = verdict
df['Winner'] = winner
df.sample(10)
#Assemble all games played in Finals and Semi-finals
finals = df[df['Round'] == 'Final']
semis = df[df['Round']== 'Semi-finals']
#Conct both dataframes resetting the index
finals_semis = pd.concat([finals, semis]).reset_index(drop=True)
finals_semis.shape
#Let's cherry-pick the columns we need
plot_df = finals_semis[['TotalGoals', 'Round', 'Year', 'month', 'day']]
plot_df[:5]
#The Plot
values2 = [68, 123, 36, 32, 65, 58, 20, 16, 6, 15, 5, 6, 25, 24, 5, 11, 11, 19, 8, 3, 7, 3, 7]
fig = go.Figure(go.Sunburst(
labels=[
"Final", "Semi-finals", "Jun", "Jul", 'Jun ', 'Jul ',
'Saturday', 'Sunday',
'Sunday ', 'Friday ', 'Saturday ', 'Tuesday ',
'Monday ', 'Saturday ', 'Tuesday ', 'Wednesday ',
'Friday ', 'Monday ', 'Sunday ', 'Saturday ', 'Thursday ', 'Tuesday ', 'Wednesday '
],
parents=[
"", "", "Final", "Final", 'Semi-finals', 'Semi-finals',
'Jun', 'Jun',
'Jul', 'Jul', 'Jul', 'Jul',
'Jun ', 'Jun ', 'Jun ', 'Jun ',
'Jul ', 'Jul ', 'Jul ', 'Jul ', 'Jul ', 'Jul ', 'Jul '
],
values=values2),
layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
)
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), title_text='Matches')
fig.data[0].marker=dict(colors=px.colors.sequential.Aggrnyl)
fig.show()
