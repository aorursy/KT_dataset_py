import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pycountry

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
cup = pd.read_csv('/kaggle/input/fifa-worldcup-2018/2018_worldcup_v3.csv')
cup.head(4)
# cup.drop(['Year', 'MatchID'],axis=1, inplace=True)
cup.rename(columns={'Home Team Name': 'Home_team',
                    'Away Team Name': 'Away_team',
                    'Home Team Goals': 'Home_goals',
                    'Away Team Goals': 'Away_goals',
                    'Assistant 1': 'Assist1',
                    'Assistant 2': 'Assist2'}, inplace=True)
cup.head(3)
cup['Hour'] = cup.Datetime.apply(lambda x: x.split(' - ')[1])
cup.Datetime = cup.Datetime.apply(lambda x: x.split(' - ')[0])
cup.head(3)
plt.figure(figsize=(10,5))
by_city = cup.groupby('City').count().reset_index()[['City','Datetime']].sort_values('Datetime', ascending=False)
sns.barplot(by_city.City, by_city.Datetime, palette='Greens_d')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.title('Number of Matches held in each Russian City', fontsize=16)
plt.xlabel('City Name', fontsize=14)
plt.ylabel('Number', fontsize=14)
plt.figure(figsize=(10,5))
games_by_hour = cup.Hour.value_counts().to_frame().reset_index()
games_by_hour.columns = ['Hour', 'Number of Mathces']
sns.barplot(games_by_hour.Hour, games_by_hour['Number of Mathces'], palette='Oranges_r')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=12)
plt.title('Number of Matches held in each Hour', fontsize=16)
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Number', fontsize=14)
# .sum().reset_index()[['Home_goals','Datetime']].sort_values('Home_goals', ascending=False)
plt.figure(figsize=(10,5))
by_city = cup.groupby('Stadium').count().reset_index()[['Stadium','Datetime']].sort_values('Datetime', ascending=False)
sns.barplot(by_city.Stadium, by_city.Datetime, palette='Blues_d')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=12)
plt.title('Number of Matches held in each Stadium', fontsize=16)
plt.xlabel('Stadium Name', fontsize=14)
plt.ylabel('Number', fontsize=14)
plt.figure(figsize=(10,7))
by_goals = cup.groupby('Datetime').sum().reset_index()[['Home_goals','Datetime']]
by_goals.Datetime = pd.to_datetime(by_goals.Datetime, format='%d %b %Y')
by_goals.Datetime = by_goals.Datetime.apply(lambda x: datetime.strftime(x, '%m/%d'))
by_goals = by_goals.sort_values('Datetime')
sns.barplot(by_goals.Home_goals, by_goals.Datetime, palette='cool_d')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=12)
plt.title('Number of Goals Scored in each Day', fontsize=16)
plt.xlabel('Number', fontsize=14)
plt.ylabel('Date', fontsize=14)
plt.figure(figsize=(10,7))
games_per_day = cup.groupby('Datetime').count().reset_index()[['Datetime', 'Stadium']].sort_values('Stadium', ascending=False)
games_per_day.Datetime = pd.to_datetime(games_per_day.Datetime, format='%d %b %Y')
games_per_day.Datetime = games_per_day.Datetime.apply(lambda x: datetime.strftime(x, '%m/%d'))
sns.barplot(games_per_day.Stadium, games_per_day.Datetime, palette='cool_d')
plt.xticks(range(5), rotation=90, fontsize=14)
plt.yticks(fontsize=12)
plt.title('Number of Mathces Held in each Day', fontsize=16)
plt.xlabel('Number', fontsize=14)
plt.ylabel('Date', fontsize=14)
cup['Total_goals'] = cup.Home_goals + cup.Away_goals
goals_by_day = cup.groupby('Datetime').Total_goals.sum().to_frame().reset_index()
goals_by_day.columns = ['Datetime2', 'Total_Goals']
goals_by_day.Datetime2 = pd.to_datetime(goals_by_day.Datetime2, format='%d %b %Y')
goals_by_day.Datetime2 = goals_by_day.Datetime2.apply(lambda x: datetime.strftime(x, '%m/%d'))
goal_ratio = pd.concat([goals_by_day, games_per_day],axis=1).drop('Datetime', axis=1)
goal_ratio.columns = ['Datetime', 'Total_Goals', 'num_of_Matches']
goal_ratio['Ratio'] = goal_ratio.Total_Goals / goal_ratio.num_of_Matches
goal_ratio = goal_ratio.sort_values('Ratio', ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(goal_ratio.Ratio, goal_ratio.Datetime, palette='cool_d')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=12)
plt.title('Avg Number of Goals Scored per Day', fontsize=16)
plt.xlabel('Number', fontsize=14)
plt.ylabel('Date', fontsize=14)
goals_by_home = cup.groupby('Home_team').sum()[['Home_goals', 'Away_goals']].reset_index()
goals_by_away = cup.groupby('Away_team').sum()[['Home_goals', 'Away_goals']].reset_index()
goals_total = pd.concat([goals_by_home, goals_by_away],axis=1)
goals_total.columns = ['Home_team','Home_Scored', 'Home_Received', 'Away_team', 'Away_Received', 'Away_Scored']
goals_total['Scored'] = goals_total.Home_Scored + goals_total.Away_Scored
goals_total['Received'] = goals_total.Home_Received + goals_total.Away_Received
goals_total = goals_total.drop(['Home_Scored', 'Home_Received', 'Away_team', 'Away_Scored', 'Away_Received'], axis=1)
goals_total['Goal_diff'] = goals_total.Scored - goals_total.Received
goals_total = goals_total.rename(columns={'Home_team': 'Team_name'})
goals_total.head(3)
plt.figure(figsize=(10,7))
goals_total = goals_total.sort_values('Scored', ascending=False)
sns.barplot(goals_total.Team_name, goals_total.Scored, palette='coolwarm_r')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(range(goals_total.Scored.max()+1), fontsize=12)
plt.title('Total Scored Goals by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Goals Number', fontsize=14)
plt.figure(figsize=(10,7))
goals_total = goals_total.sort_values('Received', ascending=False)
sns.barplot(goals_total.Team_name, goals_total.Received, palette='coolwarm')
plt.xticks(rotation=90, fontsize=14)
plt.yticks(range(goals_total.Received.max()+1),fontsize=12)
plt.title('Total Received Goals by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Goals Number', fontsize=14)
plt.figure(figsize=(10,6))
goals_total = goals_total.sort_values('Goal_diff', ascending=False).reset_index().drop(['index'], axis=1)
sns.set_style('white')
sns.barplot(goals_total.Team_name, goals_total.Goal_diff, palette='RdYlGn_r')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.yticks(range(-9, 11), fontsize=12)
plt.title('Total Goal Difference by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Goal Difference', fontsize=14)
plt.rcParams['text.latex.preview'] = True
for i in range(len(goals_total.Team_name)):
    if i in goals_total[goals_total.Goal_diff>0].index:
        plt.text(i, 0, goals_total.loc[i].Team_name, rotation=90,ha='center', va='top', fontsize=12)
    elif i in goals_total[goals_total.Goal_diff==0].index:
        plt.text(i, 0, goals_total.loc[i].Team_name, rotation=90,ha='center', va='center', fontsize=12)        
    else:
        plt.text(i, 0.15, goals_total.loc[i].Team_name, rotation=90,ha='center', va='bottom', fontsize=12)         
list_country_name = [i.name for i in list(pycountry.countries)]
def name_to_alpha(country):
    if country in list_country_name:
        return pycountry.countries.get(name=country).alpha_3
    else:
        return np.nan

def stage_group(x):
    if x[:5] == 'Group':
        return 'Group Stage'
    else:
        return x
cup['Stage_group'] = cup.Stage.apply(stage_group)
cup['Stage_group'] = pd.Categorical(cup['Stage_group'], [
    "Group Stage", "Round of 16", "Quarter-finals", "Semi-finals", "Play-off for third place","Final"])
by_attendance20 = cup[['Home_team', 'Away_team', 'Attendance', 'Stage_group']].sort_values('Attendance', ascending=False)[:20]\
.reset_index().drop('index', axis=1)

by_attendance20['alpha_3'] = by_attendance20.Home_team.apply(name_to_alpha) + ' - ' + by_attendance20.Away_team.apply(name_to_alpha)
by_attendance20.alpha_3[0] = 'RUS - SAU'
by_attendance20.alpha_3[2] = 'HRV - ENG'
by_attendance20.alpha_3[3] = 'SPN - RUS'
by_attendance20.alpha_3[8] = 'RUS - EGP'
by_attendance20.alpha_3[10] = 'BEL - ENG'
by_attendance20.alpha_3[13] = 'MAR - IRN'
by_attendance20.alpha_3[14] = 'RUS - HVR'
by_attendance20.alpha_3[17] = 'COL - ENG'

by_attendance20 = by_attendance20.rename(columns={'alpha_3':'Match'})
by_attendance20['Total_goals'] = cup.Home_goals + cup.Away_goals
plt.figure(figsize=(20,7))
by_attendance20 = by_attendance20.sort_values('Attendance', ascending=False)
sns.barplot(by_attendance20.Match, by_attendance20.Attendance, palette='Purples', hue=by_attendance20.Stage_group, edgecolor=
           sns.color_palette('Blues_d'))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.title('Total Attendance by Match', fontsize=16)
plt.xlabel('Matches', fontsize=14)
plt.ylabel('Attendance Number', fontsize=14)
plt.legend(title="Stage Group", fontsize='large', fancybox=True)