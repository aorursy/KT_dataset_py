# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns; sns.set()
import matplotlib.ticker as mtick

%matplotlib inline
#Display the table relationship diagram
img=mpimg.imread('../input/table_relationships.JPG')
imgplot = plt.imshow(img)
plt.show()
#Read in all CSV files from repository
game = pd.read_csv('../input/game.csv')
game_goalie_stats = pd.read_csv('../input/game_goalie_stats.csv')
game_plays = pd.read_csv('../input/game_plays.csv')
game_plays_players = pd.read_csv('../input/game_plays_players.csv')
game_shifts = pd.read_csv('../input/game_shifts.csv')
game_skater_stats = pd.read_csv('../input/game_skater_stats.csv')
game_teams_stats = pd.read_csv('../input/game_teams_stats.csv')
player_info = pd.read_csv('../input/player_info.csv')
team_info = pd.read_csv('../input/team_info.csv')
#create dataframe of goalie with team name and player name
goalie_team = pd.merge(game_goalie_stats, team_info, on = 'team_id')
goalie_team = pd.merge(goalie_team, player_info, on = 'player_id')
#create dataframe of player with team name and player name
skater_team = pd.merge(game_skater_stats, team_info, on = 'team_id')
skater_team = pd.merge(skater_team, player_info, on = 'player_id')
#merge the game home_team_id to the game_plays
game_plays = pd.merge(game_plays, game, on = 'game_id')
skater_name_stats = pd.merge(game, game_skater_stats, on='game_id', how='outer')
skater_name_stats = pd.merge(skater_name_stats, player_info, on='player_id', how='outer')
skater_name_stats['game_category'] = 'NA'

skater_name_stats.loc[(skater_name_stats['home_team_id'] == skater_name_stats['team_id']) & 
                   (skater_name_stats['home_goals'] > skater_name_stats['away_goals']),'game_category'] = 'Home Win'

skater_name_stats.loc[(skater_name_stats['home_team_id'] == skater_name_stats['team_id']) & 
                   (skater_name_stats['home_goals'] < skater_name_stats['away_goals']),'game_category'] = 'Home Loss'

skater_name_stats.loc[(skater_name_stats['away_team_id'] == skater_name_stats['team_id']) & 
                   (skater_name_stats['away_goals'] > skater_name_stats['home_goals']),'game_category'] = 'Away Win'

skater_name_stats.loc[(skater_name_stats['away_team_id'] == skater_name_stats['team_id']) & 
                   (skater_name_stats['home_goals'] > skater_name_stats['away_goals']),'game_category'] = 'Away Loss'
#Define how we will rank the players
skater_name_stats['rank']=skater_name_stats['goals']+skater_name_stats['assists']
#Create the list of the top 50 players
top_50_players = skater_name_stats.fillna(0)
top_50_players = top_50_players.groupby('player_id').sum()
top_50_players.sort_values('rank', ascending=False)
top_50_players = top_50_players[-50:]
top_50_players = top_50_players.reset_index()['player_id']
top_50_players = list(top_50_players)
skater_name_stats = skater_name_stats[skater_name_stats['player_id'].isin(top_50_players)]
df = skater_name_stats.groupby(['player_id','game_category']).agg({'timeOnIce':['std','mean'],
                                                        'assists':['std','mean'],
                                                       'goals':['std','mean'],
                                                       'shots':['std','mean'],
                                                        'hits':['std','mean'],
                                                        'powerPlayGoals':['std','mean'],
                                                       'powerPlayAssists':['std','mean'],
                                                       'penaltyMinutes':['std','mean'],
                                                        'penaltyMinutes':['std','mean'],
                                                       'faceOffWins':['std','mean'],
                                                       'faceoffTaken':['std','mean'],
                                                       'takeaways':['std','mean'],
                                                       'giveaways':['std','mean'],
                                                        'shortHandedGoals':['std','mean'],
                                                        'shortHandedAssists':['std','mean'],
                                                        'blocked':['std','mean'],
                                                        'plusMinus':['std','mean'],
                                                        'evenTimeOnIce':['std','mean'],
                                                        'shortHandedTimeOnIce':['std','mean'],
                                                       'powerPlayTimeOnIce':['std','mean']})
#Eliminate players without the game_category data
df = df.reset_index()
df = df[(df['game_category'] != 'NA')]
def create_boxplot_with_points(df, variable):
    plt.clf()
    sns.boxplot(x="game_category", y=(variable,'mean'), data=df.reset_index(), color='white', width=.5)
    ax = sns.swarmplot(x="game_category", y=(variable,'mean'), data=df.reset_index(), color="grey")
    ax.set_ylabel('Mean '+variable.title())
    ax.set_xlabel('')
    ax.set_title(('Mean '+variable+' compared to Game Category').title())
    #plt.ylim(0,)
    plt.savefig('/player_'+variable+'.jpg')
    pass
#Create the player plots
create_boxplot_with_points(df,'timeOnIce')
create_boxplot_with_points(df,'assists')
create_boxplot_with_points(df,'goals')
create_boxplot_with_points(df,'hits')
create_boxplot_with_points(df,'powerPlayGoals')
create_boxplot_with_points(df,'powerPlayAssists')
create_boxplot_with_points(df,'penaltyMinutes')
create_boxplot_with_points(df,'faceOffWins')
create_boxplot_with_points(df,'faceoffTaken')
create_boxplot_with_points(df,'takeaways')
create_boxplot_with_points(df,'giveaways')
create_boxplot_with_points(df,'shortHandedGoals')
create_boxplot_with_points(df,'shortHandedAssists')
create_boxplot_with_points(df,'blocked')
create_boxplot_with_points(df,'plusMinus')
create_boxplot_with_points(df,'evenTimeOnIce')
create_boxplot_with_points(df,'shortHandedTimeOnIce')
create_boxplot_with_points(df,'powerPlayTimeOnIce')
#merge the game home_team_id to the game_plays
game_and_stats = pd.merge(game, game_teams_stats, on = 'game_id', how='outer')
game_and_stats = pd.merge(game_and_stats, team_info, on = 'team_id', how='inner')
#Find stats for home wins, losses and away wins and losses by team
game_and_stats['game_category'] = 'NA'

game_and_stats.loc[(game_and_stats['home_team_id'] == game_and_stats['team_id']) & 
                   (game_and_stats['home_goals'] > game_and_stats['away_goals']),'game_category'] = 'Home Win'

game_and_stats.loc[(game_and_stats['home_team_id'] == game_and_stats['team_id']) & 
                   (game_and_stats['home_goals'] < game_and_stats['away_goals']),'game_category'] = 'Home Loss'

game_and_stats.loc[(game_and_stats['away_team_id'] == game_and_stats['team_id']) & 
                   (game_and_stats['away_goals'] > game_and_stats['home_goals']),'game_category'] = 'Away Win'

game_and_stats.loc[(game_and_stats['away_team_id'] == game_and_stats['team_id']) & 
                   (game_and_stats['home_goals'] > game_and_stats['away_goals']),'game_category'] = 'Away Loss'
df = game_and_stats.groupby(['teamName','game_category']).agg({'home_goals':['std','mean'],
                                                        'away_goals':['std','mean'],
                                                       'shots':['std','mean'],
                                                       'hits':['std','mean'],
                                                        'powerPlayOpportunities':['std','mean'],
                                                        'powerPlayGoals':['std','mean'],
                                                       'faceOffWinPercentage':['std','mean'],
                                                       'giveaways':['std','mean'],
                                                       'takeaways':['std','mean']})
def create_boxplot_with_points(df, variable):
    plt.clf()
    sns.boxplot(x="game_category", y=(variable,'mean'), data=df.reset_index(), color='white', width=.5)
    ax = sns.swarmplot(x="game_category", y=(variable,'mean'), data=df.reset_index(), color="grey")
    ax.set_ylabel('Mean '+variable.title())
    ax.set_xlabel('')
    ax.set_title(('Mean '+variable+' compared to Game Category').title())
    #plt.ylim(0,)
    plt.savefig('game_'+variable+'jpg')
    pass
#Create the game plots
create_boxplot_with_points(df,'hits')
create_boxplot_with_points(df,'shots')
create_boxplot_with_points(df,'powerPlayOpportunities')
create_boxplot_with_points(df,'powerPlayGoals')
create_boxplot_with_points(df,'giveaways')
create_boxplot_with_points(df,'takeaways')
create_boxplot_with_points(df,'faceOffWinPercentage')
create_boxplot_with_points(df,'hits')
create_boxplot_with_points(df,'shots')
create_boxplot_with_points(df,'powerPlayOpportunities')
create_boxplot_with_points(df,'powerPlayGoals')
create_boxplot_with_points(df,'giveaways')
create_boxplot_with_points(df,'takeaways')
create_boxplot_with_points(df,'faceOffWinPercentage')
corr = df.groupby('game_category').corr()
def make_mask(data):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return mask
sns.set(style = 'white')
g = sns.FacetGrid(corr.reset_index(), row = "game_category", margin_titles=True, size=8, aspect=2)
g.map_dataframe(lambda data, color: sns.heatmap(data.corr(), linewidths=0, mask=make_mask(data.corr()),annot=True))
plt.subplots_adjust(top=0.96)
g.fig.suptitle('Correlation by Game Type')
plt.savefig('heatmap_by_category.jpg')

pass
#Remove nan x positions
game_plays_position = game_plays[np.isfinite(game_plays['x'])]

#Convert a column to a datetime object
game_plays_position['dateTime'] = pd.to_datetime(game_plays_position['dateTime'])
#Create column of years from datetime Series
game_plays_position['year'] = game_plays_position['dateTime'].dt.year
#Free up some memory
del game_shifts
del game_plays
del game_plays_players
#lets only look at hits
game_plays_goals = game_plays_position[game_plays_position['event'] == 'Goal'] 
game_plays_hits = game_plays_position[game_plays_position['event'] == 'Hit']
#Seems like there is some issue with data type per some reading online.  Needs to be float not an object
cols = ['x', 'y']
game_plays_position[cols] = game_plays_position[cols].astype(float)
season_scores = game_plays_position.groupby(['year', 'game_id'])[['goals_home', 'goals_away']].max()
season_scores.groupby('year')[['goals_home', 'goals_away']].mean()
#Let's look at where all the Goals have occurred on the ice during regular game play time only (periods 1 - 3)
reg_time = game_plays_goals[game_plays_goals['period'] <=3]

#This took forever to run on my computer but eventually showed up!
sns.set(style = 'white')
g = sns.FacetGrid(reg_time, col = "year", row = 'period', margin_titles=True)
g.map(sns.kdeplot, 'x', 'y')

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Player Position When Scoring Regular time')
plt.savefig('kde_position_plot.jpg')
#Add team name to scoring team goals
goals_by_team = pd.merge(game_plays_goals, team_info, how='inner', left_on='team_id_for', right_on='team_id')

#Only keep the max number of goals (game ending scores)
goals_by_team = goals_by_team.groupby(['teamName','game_id']).max()

#Look at mean goal differential as well as plus 1 standard deviation and minus 1 standard deviation
goals_by_team_summarized = goals_by_team.groupby('teamName').agg({'goals_home':['std','mean'],
                                                                   'goals_away':['std','mean']})
goals_by_team_summarized['goal_differential'] = goals_by_team_summarized['goals_home']['mean'] - goals_by_team_summarized['goals_away']['mean']

goals_by_team_summarized['goal_differential_p1std'] = (goals_by_team_summarized['goals_home']['mean'] + goals_by_team_summarized['goals_home']['std']) - \
    (goals_by_team_summarized['goals_away']['mean'] + goals_by_team_summarized['goals_away']['std'])

goals_by_team_summarized['goal_differential_m1std'] = (goals_by_team_summarized['goals_home']['mean'] - goals_by_team_summarized['goals_home']['std']) - \
    (goals_by_team_summarized['goals_away']['mean'] - goals_by_team_summarized['goals_away']['std'])

#Sort by goal differential
goals_by_team_summarized = goals_by_team_summarized.sort_values('goal_differential', ascending=False)
goals_by_team_summarized = goals_by_team_summarized[['goal_differential_m1std','goal_differential','goal_differential_p1std']]
#Create the long version dataframe to see a line for each team
df = goals_by_team_summarized.reset_index()
df_long = df.melt(id_vars=['teamName'], var_name='goal_differential')

#Create a plot of team & goal differentials
sns.set(style = 'white')
ax = sns.pointplot(x="value", y="teamName", hue = "teamName", data=df_long, join=False, size=8, aspect = 2)
ax.set(xlabel='Home vs Away Goal Differential', ylabel='Team Name', title = 'Home Team Advantage')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('goal_differential_by_team.jpg')
pass
#Add team name to scoring team goals
goals_by_team = pd.merge(game_plays_goals, team_info, how='inner', left_on='team_id_for', right_on='team_id')

#Only keep the max number of goals (game ending scores)
goals_by_team = goals_by_team.groupby(['teamName','game_id']).max()

#Look at mean goal differential as well as plus 1 standard deviation and minus 1 standard deviation
goals_by_team_summarized = goals_by_team.groupby(['teamName','year']).agg({'goals_home':['std','mean'],
                                                                   'goals_away':['std','mean']})

goals_by_team_summarized['goal_differential'] = goals_by_team_summarized['goals_home']['mean'] - goals_by_team_summarized['goals_away']['mean']

goals_by_team_summarized['goal_differential_p1std'] = (goals_by_team_summarized['goals_home']['mean'] + goals_by_team_summarized['goals_home']['std']) - \
    (goals_by_team_summarized['goals_away']['mean'] + goals_by_team_summarized['goals_away']['std'])

goals_by_team_summarized['goal_differential_m1std'] = (goals_by_team_summarized['goals_home']['mean'] - goals_by_team_summarized['goals_home']['std']) - \
    (goals_by_team_summarized['goals_away']['mean'] - goals_by_team_summarized['goals_away']['std'])

#Sort by goal differential
goals_by_team_summarized = goals_by_team_summarized.sort_values('goal_differential', ascending=False)
goals_by_team_summarized = goals_by_team_summarized[['goal_differential_m1std','goal_differential','goal_differential_p1std']]
#Create the long version dataframe to see a line for each team
df = goals_by_team_summarized.reset_index()
df_long = df.melt(id_vars=['teamName','year'], var_name='var_name')
sns.set(style = 'white')
sns.set_context("paper") 
g = sns.FacetGrid(df_long, col = "year", margin_titles=True, size=6, aspect=2, sharey=False, sharex=False, col_wrap=3)
g.map(sns.pointplot, 'value', 'teamName')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Home Team Advantage by Season')
axes = g.axes
for plot in axes:
    plot.set_xlim(-0.5,1.5)
plt.savefig('team_advantage_by_season.jpg')
pass
sns.set(style = 'white')
g = sns.FacetGrid(df_long, col = "teamName", col_wrap=6, margin_titles=True, size=3, aspect=2)
g.map(sns.pointplot, 'year', 'value')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Home Team Advantage by Season')
plt.savefig('goal_differential_by_season.jpg')
pass
#Penguins Advantage by Year
sns.boxplot(x="year", y="value",
            hue="teamName",
            data=df_long[(df_long['teamName']=='Penguins') | (df_long['teamName']=='Bruins') | (df_long['teamName']=='Blackhawks')])
plt.savefig('pens_bruins_blackhawks.jpg')
pass
#We will now look at differences in time zones
venue_by_team = pd.merge(game_plays_goals, team_info, how='inner', left_on='home_team_id', right_on='team_id')
#venue_by_team = pd.merge(venue_by_team, team_info, how='inner', left_on='home_team_id', right_on='team_id')

#Add in the venue of the away team as well
goals_by_team = pd.merge(game_plays_goals, team_info, how='inner', left_on='away_team_id', right_on='team_id')
#Find the venue for each team
venue_by_team = venue_by_team.groupby(['teamName','venue_time_zone_offset']).sum().reset_index()[['teamName','venue_time_zone_offset']]
goals_by_team = pd.merge(goals_by_team, venue_by_team, left_on='teamName', right_on='teamName')
goals_by_team['timezone_diff'] = goals_by_team['venue_time_zone_offset_x'] - goals_by_team['venue_time_zone_offset_y']
#Only keep the max number of goals (game ending scores)
goals_by_team = goals_by_team.groupby(['teamName','game_id']).max()

#Look at mean goal differential as well as plus 1 standard deviation and minus 1 standard deviation
goals_by_team_summarized = goals_by_team.groupby(['timezone_diff','year']).agg({'goals_home':['std','mean'],
                                                                   'goals_away':['std','mean']})

goals_by_team_summarized['goal_differential'] = goals_by_team_summarized['goals_home']['mean'] - goals_by_team_summarized['goals_away']['mean']

goals_by_team_summarized['goal_differential_p1std'] = (goals_by_team_summarized['goals_home']['mean'] + goals_by_team_summarized['goals_home']['std']) - \
    (goals_by_team_summarized['goals_away']['mean'] + goals_by_team_summarized['goals_away']['std'])

goals_by_team_summarized['goal_differential_m1std'] = (goals_by_team_summarized['goals_home']['mean'] - goals_by_team_summarized['goals_home']['std']) - \
    (goals_by_team_summarized['goals_away']['mean'] - goals_by_team_summarized['goals_away']['std'])

#Sort by goal differential
goals_by_team_summarized = goals_by_team_summarized.sort_values('goal_differential', ascending=False)
goals_by_team_summarized = goals_by_team_summarized[['goal_differential_m1std','goal_differential','goal_differential_p1std']]
#Create the long version dataframe to see a line for each team
df = goals_by_team_summarized.reset_index()
df_long = df.melt(id_vars=['timezone_diff','year'], var_name='var_name')
sns.set(style = 'white')
g = sns.FacetGrid(df_long, col = "timezone_diff", margin_titles=True, col_wrap=3,size=4, aspect=2)
g.map(sns.pointplot, 'year', 'value')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Home Team Advantage by Time Zone Difference')
plt.savefig('goal_differential_by_time_zone_diff.jpg')
pass
home_goals_by_team = goals_by_team[goals_by_team['home_team_id']==goals_by_team['team_id_for']]
away_game_plays_goals = goals_by_team[goals_by_team['away_team_id']==goals_by_team['team_id_for']]
#Split into home and away goals
home_game_plays_goals = game_plays_goals[game_plays_goals['home_team_id']==game_plays_goals['team_id_for']]
away_game_plays_goals = game_plays_goals[game_plays_goals['away_team_id']==game_plays_goals['team_id_for']]
home_game_plays_goals.head()
#Let's look at where all the Goals have occurred on the ice FOR HOME TEAM during regular play
home_reg_time = home_game_plays_goals[game_plays_goals['period'] <=3]

#This took forever to run on my computer but eventually showed up!
sns.set(style = 'white')
g = sns.FacetGrid(home_reg_time, col = "year", row = 'period', margin_titles=True)
g.map(sns.kdeplot, 'x', 'y')

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Home Player Position When Scoring Regular time')
plt.savefig('player_position_regular_scoring_kdeplot.jpg')
pass
#Let's look at where all the Goals have occurred on the ice FOR AWAY TEAM
away_reg_time = away_game_plays_goals[game_plays_goals['period'] <=3]

#This took forever to run on my computer but eventually showed up!
sns.set(style = 'white')
g = sns.FacetGrid(away_reg_time, col = "year", row = 'period', margin_titles=True)
g.map(sns.kdeplot, 'x', 'y')

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Away Player Position When Scoring Regular time')
plt.savefig('away_player_position_regular_scoring_kdeplot.jpg')
pass
