import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-bright')
fifa20 = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
fifa20.head()
fifa20.shape
print('Total number of countries', fifa20.nationality.nunique())
print('total number of clubs', fifa20.club.nunique())
print('Maximum number of players from one club', fifa20.club.value_counts().max())
print('Countries with most number of players \n', fifa20.nationality.value_counts().head(10))
fifa20.drop(columns=['sofifa_id','player_url','body_type','real_face','loaned_from','player_tags','player_traits'],inplace=True)
print('Maximum Overall Rating: ' +str(fifa20.loc[fifa20['overall'].idxmax()][0]))
print('Maximum Potential: ' +str(fifa20.loc[fifa20['potential'].idxmax()][0]))
print('Most Valuable Player: ' +str(fifa20.loc[fifa20['value_eur'].idxmax()][0]))
print('Most Earning Player: ' +str(fifa20.loc[fifa20['wage_eur'].idxmax()][0]))
print('Fastest player: ' +str(fifa20.loc[fifa20['pace'].idxmax()][0]))
print('Best Shooting: ' +str(fifa20.loc[fifa20['shooting'].idxmax()][0]))
print('Best Passing: ' +str(fifa20.loc[fifa20['passing'].idxmax()][0]))
print('Best Dribbling: ' +str(fifa20.loc[fifa20['dribbling'].idxmax()][0]))
print('Best Defending: ' +str(fifa20.loc[fifa20['defending'].idxmax()][0]))
print('Best Physical: ' +str(fifa20.loc[fifa20['physic'].idxmax()][0]))

print('Best Attacking crossing: ' +str(fifa20.loc[fifa20['attacking_crossing'].idxmax()][0]))
print('Best Attacking finishing: ' +str(fifa20.loc[fifa20['attacking_finishing'].idxmax()][0]))
print('Best Attacking heading accuracy: ' +str(fifa20.loc[fifa20['attacking_heading_accuracy'].idxmax()][0]))
print('Best Attacking short passing: ' +str(fifa20.loc[fifa20['attacking_short_passing'].idxmax()][0]))
print('Best Attacking volleys: ' +str(fifa20.loc[fifa20['attacking_volleys'].idxmax()][0]))
print('Best Skill dribbling: ' +str(fifa20.loc[fifa20['skill_dribbling'].idxmax()][0]))
print('Best Curve: ' +str(fifa20.loc[fifa20['skill_curve'].idxmax()][0]))
print('Best Freekick accuracy: ' +str(fifa20.loc[fifa20['skill_fk_accuracy'].idxmax()][0]))
print('Best Long passing: ' +str(fifa20.loc[fifa20['skill_long_passing'].idxmax()][0]))
print('Best Ball control: ' +str(fifa20.loc[fifa20['skill_ball_control'].idxmax()][0]))
print('Best Movement Acceleration: ' +str(fifa20.loc[fifa20['movement_acceleration'].idxmax()][0]))
print('Best Sprint speed: ' +str(fifa20.loc[fifa20['movement_sprint_speed'].idxmax()][0]))
print('Best Agility: ' +str(fifa20.loc[fifa20['movement_agility'].idxmax()][0]))
print('Best Reactions: ' +str(fifa20.loc[fifa20['movement_reactions'].idxmax()][0]))
print('Best Balance: ' +str(fifa20.loc[fifa20['movement_balance'].idxmax()][0]))
print('Best Shot power: ' +str(fifa20.loc[fifa20['power_shot_power'].idxmax()][0]))
print('Best Jumping: ' +str(fifa20.loc[fifa20['power_jumping'].idxmax()][0]))
print('Best Stamina: ' +str(fifa20.loc[fifa20['power_stamina'].idxmax()][0]))
print('Best Strength: ' +str(fifa20.loc[fifa20['power_strength'].idxmax()][0]))
print('Best Long shots: ' +str(fifa20.loc[fifa20['power_long_shots'].idxmax()][0]))
print('Best Aggression: ' +str(fifa20.loc[fifa20['mentality_aggression'].idxmax()][0]))
print('Best Interceptions: ' +str(fifa20.loc[fifa20['mentality_interceptions'].idxmax()][0]))
print('Best Positioning: ' +str(fifa20.loc[fifa20['mentality_positioning'].idxmax()][0]))
print('Best Vision: ' +str(fifa20.loc[fifa20['mentality_vision'].idxmax()][0]))
print('Best Penalties: ' +str(fifa20.loc[fifa20['mentality_penalties'].idxmax()][0]))
print('Best Composure: ' +str(fifa20.loc[fifa20['mentality_composure'].idxmax()][0]))
print('Best Marking: ' +str(fifa20.loc[fifa20['defending_marking'].idxmax()][0]))
print('Best Standing tackle: ' +str(fifa20.loc[fifa20['defending_standing_tackle'].idxmax()][0]))
print('Best Sliding tackle: ' +str(fifa20.loc[fifa20['defending_sliding_tackle'].idxmax()][0]))
print('Best Diving: ' +str(fifa20.loc[fifa20['goalkeeping_diving'].idxmax()][0]))
print('Best Handling: ' +str(fifa20.loc[fifa20['goalkeeping_handling'].idxmax()][0]))
print('Best Kicking: ' +str(fifa20.loc[fifa20['goalkeeping_kicking'].idxmax()][0]))
print('Best Positioning: ' +str(fifa20.loc[fifa20['goalkeeping_positioning'].idxmax()][0]))
print('Best Reflexes: ' +str(fifa20.loc[fifa20['goalkeeping_reflexes'].idxmax()][0]))
def nation(x=None):
    if x == None:
        return fifa20[['short_name','age','club','nationality','overall','player_positions','contract_valid_until']]
    else:
        return fifa20[fifa20['nationality'] == x][['short_name','age','club','nationality','overall','player_positions','contract_valid_until']]

nation('India')
nation('India').shape
print('The average age of Indian players is:', nation('India').age.mean() )
print('The average age of all players in the entire database:', nation().age.mean() )
print('Youngest player in Indian Team is ', nation('India').age.min())
print('Youngest players in entire Database are\n',nation().age.sort_values().head(10).reset_index(drop=True))
def club(x):
    return fifa20[fifa20['club'] == x][['short_name','age','club','nationality','overall','player_positions',
       'contract_valid_until']]
club('Liverpool').head(21).reset_index(drop=True)
club('Real Madrid').head(20).reset_index(drop=True)
fifa20.isnull().sum().sort_values(ascending=False).head(50)
fifa20['nation_position'].fillna('None yet',inplace = True)
fifa20['dribbling'].fillna(fifa20['dribbling'].mean(),inplace=True)
fifa20['shooting'].fillna(fifa20['shooting'].mean(),inplace=True)
fifa20['pace'].fillna(fifa20['pace'].mean(),inplace=True)
fifa20['passing'].fillna(fifa20['passing'].mean(),inplace=True)
fifa20['physic'].fillna(fifa20['physic'].mean(),inplace=True)
fifa20['defending'].fillna(fifa20['defending'].mean(),inplace=True)
fifa20['release_clause_eur'].fillna('€250K',inplace=True)
fifa20['joined'].fillna('2018-07-01',inplace=True)
fifa20['contract_valid_until'].fillna('2019',inplace=True)
fifa20['team_position'].fillna('None yet',inplace=True)
#We will fill '00' for the rest of the numerical values.
fifa20.fillna('00', inplace=True)
fifa20.head()
def nation_stats(x):
    age= print('The average age of 20 member squad is', nation(x).head(20).age.mean())
    overall = print('Average rating of 20 member squad is', nation(x).head(20).overall.mean())
    return age, overall
nation('Spain').head(20)
nation_stats('Spain')
nation_stats('Germany')
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
ax = sns.countplot('preferred_foot', data=fifa20,palette='bone')
ax.set_xlabel(xlabel = 'Preffered foot', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Preffered foot of players', fontsize = 20)
plt.show()
plt.figure(figsize=(18,8))
#without_subs = fifa20[np.logical_or(fifa20['team_position'] != 'SUB',fifa20['team_position'] != 'RES',fifa20['team_position'] != 'None yet')]
#without_RES = without_subs[without_subs['team_position'] != 'RES' ]
#without_subs=fifa20.drop(fifa20[(fifa20.team_position = 'SUB') | (fifa20.team_position = 'RES')|(fifa20.team_position = 'None yet')].index)
without_subs = fifa20[~fifa20.team_position.isin(['SUB','RES','None yet'])]
ax = sns.countplot('team_position', data=without_subs,palette='bone')
ax.set_xlabel(xlabel = 'Player Positions', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Number of Player Positions', fontsize = 20)
plt.show()
plt.figure(figsize = (13, 10))
weight = fifa20[(fifa20.weight_kg < 100 ) & (fifa20.weight_kg > 60)]
ax = sns.countplot(x = 'weight_kg', data = weight, palette = 'dark')
ax.set_title(label = 'Count of players on Basis of Weight', fontsize = 20)
ax.set_xlabel(xlabel = 'Weight in kg', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()
plt.figure(figsize = (13, 10))
height = fifa20[(fifa20.height_cm > 165) & (fifa20.height_cm < 195)]
ax = sns.countplot(x = 'height_cm', data = height, palette = 'dark')
ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)
ax.set_xlabel(xlabel = 'Height in cm', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
ax = sns.countplot('age', data=fifa20,palette='bone')
ax.set_xlabel(xlabel = 'Age of Players', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Age distribution of players', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
ax = sns.countplot('skill_moves', data=fifa20,palette='bone')
ax.set_xlabel(xlabel = 'Number of skill moves', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Skill moves of players', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
ax = sns.countplot('work_rate', data=fifa20,palette='bone')
ax.set_xlabel(xlabel = 'Work Rate', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Work Rate of players', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
ax = sns.countplot('weak_foot', data=fifa20,palette='bone')
ax.set_xlabel(xlabel = 'Weak foot players rating', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Rating with Weak foot', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
rating = fifa20[fifa20.overall > 80]
ax = sns.countplot('overall', data=rating,palette='bone')
ax.set_xlabel(xlabel = 'Players rating', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Players with 80 above rating', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
rating = fifa20[fifa20.overall >= 90]
ax = sns.countplot('overall', data=rating,palette='bone')
ax.set_xlabel(xlabel = 'Players rating', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Players with 90 and above rating', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
pot = fifa20[fifa20.potential > 80]
ax = sns.countplot('potential', data = pot ,palette='bone')
ax.set_xlabel(xlabel = 'Players Potential', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Players with 80 above Potential', fontsize = 20)
plt.show()
plt.figure(figsize=(10,8))
plt.style.use('seaborn-bright')
pot = fifa20[fifa20.potential > 89]
ax = sns.countplot('potential', data = pot ,palette='bone')
ax.set_xlabel(xlabel = 'Players Potential', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Players with 90 and above Potential', fontsize = 20)
plt.show()
sns.set(style ="dark", palette="colorblind")
plt.figure(figsize=(12,8))
plt.style.use('ggplot')
ax = sns.distplot(fifa20.overall,bins=50,kde=False)
ax.set_xlabel(xlabel = 'Players rating', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Rating distribution of players', fontsize = 20)
plt.show()
sns.set(style="dark",palette="colorblind")
plt.style.use('ggplot')
plt.figure(figsize=(12,8))
ax = sns.distplot(fifa20.potential,bins=50,kde=False)
ax.set_xlabel(xlabel = 'Players Potential', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Potential distribution of players', fontsize = 20)
plt.show()
sns.set(style="dark",palette="colorblind")
plt.style.use('ggplot')
plt.figure(figsize=(12,8))
ax = sns.distplot(fifa20.age,bins=25,kde=False)
ax.set_xlabel(xlabel = 'Players age', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Age distribution of players', fontsize = 20)
plt.show()
plt.figure(figsize=(12,8))
sns.lmplot(data = fifa20, x = 'age', y = 'pace',lowess=True,scatter_kws={'alpha':0.1, 's':5,'color':'green'}, 
           line_kws={'color':'red'})

plt.figure(figsize=(12,8))
sns.lmplot(data = fifa20, x = 'age', y = 'overall',lowess=True,scatter_kws={'alpha':0.1, 's':5,'color':'green'}, 
           line_kws={'color':'red'})
fifa20.iloc[fifa20.groupby(fifa20.team_position)['overall'].idxmax()][['short_name','age','club','nationality','overall','team_position']].sort_values(by='age').reset_index(drop=True)

fifa20.iloc[fifa20.groupby(fifa20.team_position)['potential'].idxmax()][['short_name','age','club','nationality','potential','team_position']].sort_values(by='age').reset_index(drop=True)

plt.figure(figsize=(14,8))
sns.set(style='whitegrid',palette='colorblind')
plt.style.use('ggplot')
major_clubs = ('Real Madrid','FC Barcelona','Manchester United','Paris Saint-Germain', 'Manchester City', 'Liverpool','Chelsea','Juventus','Ajax','Borussia Dortmund','FC Bayern München')
data_clubs = fifa20.loc[fifa20.club.isin(major_clubs) & fifa20.age]
ax = sns.boxplot(x= data_clubs.club, y=data_clubs.age, data=data_clubs)
ax.set_xlabel(xlabel='Clubs', fontsize=16)
ax.set_ylabel(ylabel='Age Distribution of Players', fontsize=16)
ax.set_title(label='Age distribution of players among Top Clubs', fontsize=20)
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize=(14,8))
sns.set(style='whitegrid',palette='colorblind')
plt.style.use('ggplot')
major_clubs = ('Real Madrid','FC Barcelona','Manchester United','Paris Saint-Germain', 'Manchester City', 'Liverpool','Chelsea','Juventus','Ajax','Borussia Dortmund','FC Bayern München')
overall_clubs = fifa20.loc[fifa20.club.isin(major_clubs) & fifa20.overall]
ax = sns.boxplot(x='club',y='overall',data=overall_clubs)
ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
ax.set_ylabel(ylabel='Overall Rating of Players',fontsize=16)
ax.set_title(label='Overall Rating of players among Top Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(14,8))
sns.set(style='whitegrid',palette='deep')
plt.style.use('ggplot')

ax = sns.boxplot(x='club',y='wage_eur',data=data_clubs)
ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
ax.set_ylabel(ylabel='Wages of Players',fontsize=16)
ax.set_title(label='Wage Distribution of Players among Top Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(14,8))
sns.set(style='whitegrid',palette='dark')
plt.style.use('ggplot')

ax = sns.boxplot(x='club',y='value_eur',data=data_clubs)
ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
ax.set_ylabel(ylabel='Value of Players',fontsize=16)
ax.set_title(label='Value of Players among Top Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(14,8))
#sns.set(style='whitegrid',palette='dark')
plt.style.use('ggplot')

ax = sns.violinplot(x='club',y='international_reputation',data=data_clubs)
ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
ax.set_ylabel(ylabel='International Reputation of Players',fontsize=16)
ax.set_title(label='International Reputation of Players in Top Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(14,8))
#sns.set(style='whitegrid',palette='dark')
plt.style.use('ggplot')

ax = sns.violinplot(x='club',y='pace',data=data_clubs)
ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
ax.set_ylabel(ylabel='Pace of Players',fontsize=16)
ax.set_title(label='Pace of Players in Top Clubs',fontsize=20)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(14,8))
#sns.set(style='whitegrid',palette='dark')
plt.style.use('seaborn-bright')

ax = sns.jointplot(x=fifa20.age,y=fifa20.potential,joint_kws={'alpha':0.5,'s':10,'color':'red'}, marginal_kws={'color':'red'})
#ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
#ax.set_ylabel(ylabel='Pace of Players',fontsize=16)
#ax.set_title(label='Pace of Players in Top Clubs',fontsize=20)
#plt.xticks(rotation=90)
plt.show()
#plt.figure(figsize=(14,8))
#sns.set(style='whitegrid',palette='dark')
plt.style.use('seaborn-bright')

ax = sns.jointplot(x=fifa20.age,y=fifa20.wage_eur,joint_kws={'alpha':0.5,'s':10,'color':'red'}, marginal_kws={'color':'red'})
#ax.set_xlabel(xlabel='Major Clubs',fontsize=16)
#ax.set_ylabel(ylabel='Pace of Players',fontsize=16)
#ax.set_title(label='Pace of Players in Top Clubs',fontsize=20)
#plt.xticks(rotation=90)
plt.show()
sns.lmplot(x='dribbling', y='passing', data=fifa20, col='preferred_foot',scatter_kws = {'alpha':0.1,'color':'navy'},
           line_kws={'color':'red'})
sns.lmplot(x='dribbling', y='skill_ball_control', data=fifa20, col='preferred_foot',scatter_kws = {'alpha':0.1,'color':'navy'},
           line_kws={'color':'red'})
sns.jointplot(x='defending', y='physic', data=fifa20,joint_kws={'alpha':0.1,'s':10,'color':'navy'}, marginal_kws={'color':'navy'} )
fifa20[fifa20.age < 20 ][['short_name','age','club','nationality','overall','potential','player_positions']].head(10).reset_index(drop=True)
fifa20[fifa20.age < 30 ][['short_name','age','club','nationality','overall','potential','player_positions']].head(10).reset_index(drop=True)
fifa20[fifa20.age >= 30 ][['short_name','age','club','nationality','overall','potential','player_positions']].head(10).reset_index(drop=True)
fifa20[fifa20['preferred_foot'] == 'Left' ][['short_name','age','club','nationality','overall','potential','player_positions']].head(10).reset_index(drop=True)
fifa20[fifa20['preferred_foot'] == 'Right' ][['short_name','age','club','nationality','overall','potential','player_positions']].head(10).reset_index(drop=True)