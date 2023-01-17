import numpy as np
import pandas as pd
import random as rnd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

# This function is written for good visualization of the graphs and its called every time when we create a plot

# If there is a way to declare all the parameters globally help would be much appriciated

def style():
    plt.subplots(figsize=(15,9))
    sns.set_style("whitegrid")
    sns.despine()
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    
deliveries = pd.read_csv('../input/deliveries.csv')
matches = pd.read_csv('../input/matches.csv')
matches.head() # The head function displays the top 5 rows from the dataset
matches.columns
deliveries.head()
deliveries.columns
matches.isnull().sum()
matches.drop('umpire3',axis = 1, inplace=True)
deliveries.isnull().sum()
matches['team1'].unique()
matches.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
       'Rising Pune Supergiant', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace = True)

deliveries.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
       'Rising Pune Supergiant', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],['SR','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace = True)

# lets see how many matches are being played every season
style()
sns.countplot(x = 'season', data = matches)

# By this graph you ca nsee that Mumbai Indians have won most of the matches in the IPL
style()
sns.countplot(x = 'winner', data = matches, palette = ['darkorange','#d11d9b','purple',
                                                       'tomato','gold','royalblue','red','#e04f16','yellow','gold'
                                                       ,'black','silver','b'])

# Top cities where the matches are held
style()
fav_cities = matches['city'].value_counts().reset_index()
fav_cities.columns = ['city','count']
sns.barplot(x = 'count',y = 'city', data = fav_cities[:10])

# overall scores of every season
style()
batsmen = matches[['id','season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
season = batsmen.groupby(['season'])['total_runs'].sum().reset_index()
sns.barplot(x = 'season', y = 'total_runs', data = season,palette='spring')

style()
fav_ground = matches['venue'].value_counts().reset_index()
fav_ground.columns = ['venue','count']
sns.barplot(x = 'count',y = 'venue', data = fav_ground[:10], palette = 'Reds')

orange_cap = matches[['id','season']]
orange_cap = orange_cap.merge(deliveries,left_on = 'id', right_on = 'match_id')
orange_cap = orange_cap.groupby(['batsman','season'])['batsman_runs'].sum().reset_index()
orange_cap = orange_cap.sort_values('batsman_runs',ascending=False)
orange_cap = orange_cap.drop_duplicates(subset = ['season'],keep = 'first')
style()
sns.barplot(x = 'season', y = 'batsman_runs', data = orange_cap,palette= 'Oranges')
types_of_dismissal = [ 'caught', 'bowled', 'lbw', 'caught and bowled','stumped',  'hit wicket']
purple_cap = deliveries[deliveries['dismissal_kind'].isin(types_of_dismissal)]
purple_cap = purple_cap.merge(matches,left_on='match_id', right_on = 'id')
purple_cap = purple_cap.groupby(['season','bowler'])['dismissal_kind'].count().reset_index()
purple_cap = purple_cap.sort_values('dismissal_kind',ascending = False)
purple_cap = purple_cap.drop_duplicates('season',keep = 'first').sort_values(by='season')
purple_cap.columns=['season','bowler','count_wickets']
style()
sns.barplot(x = 'season', y = 'count_wickets', data = purple_cap, palette = 'RdPu')

# each team with is max boundaries in the entire IPL
sixes = deliveries[deliveries['batsman_runs'] == 6]['batting_team'].value_counts().reset_index()
fours = deliveries[deliveries['batsman_runs'] == 4]['batting_team'].value_counts().reset_index()
scores = sixes.merge(fours,left_on = 'index', right_on = 'index')
scores.columns = [['team_name','4s','6s']]
sns.set()
from matplotlib.colors import ListedColormap
scores.set_index('team_name').plot(kind = 'bar',stacked = True, colormap=ListedColormap(sns.color_palette("GnBu", 10)),  figsize=(15,6))

# Number of times each team went to finals and won the match
final_list = matches.drop_duplicates(subset=['season'],keep = 'last')
team_names = pd.concat([final_list['team1'],final_list['team2']]).value_counts().reset_index()
team_names.columns = ['team_name','count']
final_winners_count = final_list['winner'].value_counts().reset_index()
team_names = team_names.merge(final_winners_count,left_on = 'team_name',right_on = 'index', how = 'outer')
team_names.drop(['index'],inplace = True,axis = 1)
team_names['winner'].fillna(0,inplace = True)
sns.set()
team_names.set_index('team_name').plot(kind = 'bar',colormap=ListedColormap(sns.color_palette("summer", 10)),  figsize=(15,6))


ump = pd.DataFrame(pd.concat([matches['umpire1'], matches['umpire2']]),columns = ['count'])
ump = ump.apply(pd.value_counts)
style()
sns.barplot(y = ump[:10].index, x= ump[:10]['count'], palette = 'winter')

MOTM = matches['player_of_match'].value_counts()
style()
sns.barplot(x = MOTM.index[:10], y = MOTM.values[:10])
df=matches[matches['toss_winner']==matches['winner']]
slices=[len(df),(577-len(df))]
labels=['yes','no']
plt.pie(slices,labels=labels)
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.show()
# player with most boundries
data = deliveries[(deliveries['batsman_runs'] == 4) | (deliveries['batsman_runs'] == 6)][['batsman','batsman_runs']].groupby('batsman').count().reset_index().sort_values(ascending = False, by = 'batsman_runs')
plt.subplots(figsize=(15,9))
sns.set_style("whitegrid")
sns.despine()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.barplot(x = 'batsman_runs', y = 'batsman', data = data[:10],palette="Blues_d")
