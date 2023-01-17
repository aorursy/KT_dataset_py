import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
mypal = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Grab the color pal
import os
import gc

MENS_DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament'
WOMENS_DIR = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament'
MTeams = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MTeams.csv')
MTeams.sort_values('FirstD1Season', ascending=False).head(5)
# Womens' data does not contain years joined :(
WTeams = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WTeams.csv')
WTeams.head()
MSeasons = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MSeasons.csv')
WSeasons = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WSeasons.csv')
MSeasons.head()
MRegularSeasonCompactResults = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
WRegularSeasonCompactResults = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')
MNCAATourneyCompactResults = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
WNCAATourneyCompactResults = pd.read_csv(f'{WOMENS_DIR}/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')
MAllResults = MRegularSeasonCompactResults.append(MNCAATourneyCompactResults)
WAllResults = WRegularSeasonCompactResults.append(WNCAATourneyCompactResults)
MAllResults.head()
# Lets Add the winning and losing team names to the results
MAllResults = \
    MAllResults \
    .merge(MTeams[['TeamName', 'TeamID']],
           left_on='WTeamID',
           right_on='TeamID',
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'WTeamName'}) \
    .merge(MTeams[['TeamName', 'TeamID']],
           left_on='LTeamID',
           right_on='TeamID') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'LTeamName'})

WAllResults = \
    WAllResults \
    .merge(WTeams[['TeamName', 'TeamID']],
           left_on='WTeamID',
           right_on='TeamID',
           validate='many_to_one') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'WTeamName'}) \
    .merge(WTeams[['TeamName', 'TeamID']],
           left_on='LTeamID',
           right_on='TeamID') \
    .drop('TeamID', axis=1) \
    .rename(columns={'TeamName': 'LTeamName'})
WAllResults.head()
WAllResults['TScore'] = WAllResults['WScore'] + WAllResults['LScore']
WScores = WAllResults.groupby('Season')['TScore'].sum() / WAllResults.groupby('Season')['TScore'].count()
MAllResults['TScore'] = MAllResults['WScore'] + MAllResults['LScore']
MScores = MAllResults.groupby('Season')['TScore'].sum() / MAllResults.groupby('Season')['TScore'].count()
plt.plot(WScores, 'red', label = 'Women')
plt.plot(MScores, 'blue', label = 'Men')
plt.title('Points scored per game')
leg=plt.legend(loc='best')
def build_stats(df):
    df['DiffScore'] = df['WScore'] - df['LScore']
    df['counter'] = 1
    games_won = df.groupby('WTeamName')['counter'].count()
    games_lost = df.groupby('LTeamName')['counter'].count()
    games_won_2015 = df[df['Season'] >= 2015].groupby('WTeamName')['counter'].count()
    games_lost_2015 = df[df['Season'] >= 2015].groupby('LTeamName')['counter'].count()
    Points_scored_won = df.groupby('WTeamName')['WScore'].sum()
    Points_scored_loss = df.groupby('LTeamName')['LScore'].sum()
    Points_scored_against_won = df.groupby('WTeamName')['LScore'].sum()
    Points_scored_against_loss = df.groupby('LTeamName')['WScore'].sum()
    Diff_W = df.groupby('WTeamName')['DiffScore'].mean()
    Diff_L = - df.groupby('LTeamName')['DiffScore'].mean()
    games_played = pd.concat([games_won, games_lost, games_won_2015, games_lost_2015, Diff_W, Diff_L, Points_scored_won, Points_scored_loss, Points_scored_against_won, Points_scored_against_loss], axis = 1, sort = True)
    games_played.columns = ['Games_W', 'Games_L', 'Games_W_2015', 'Games_L_2015', 'Diff_W', 'Diff_L', 'Points_scored_W','Points_scored_L','Points_scored_against_W','Points_scored_against_L',]
    
    games_played['Games_Total'] = games_played['Games_L'] + games_played['Games_W']
    games_played['Games_Total_2015'] = games_played['Games_L_2015'] + games_played['Games_W_2015']
    games_played['Winning_Rate'] = games_played['Games_W'] * 100 / games_played['Games_Total']
    games_played['Avg_Diff'] = (games_played['Diff_W'] * games_played['Games_W'] + games_played['Diff_L'] * games_played['Games_L']) / games_played['Games_Total']
    games_played['Avg_Points_scored'] = (games_played['Points_scored_W'] + games_played['Points_scored_L']) / games_played['Games_Total']
    games_played['Avg_Points_scored_against'] = (games_played['Points_scored_against_W'] + games_played['Points_scored_against_L']) / games_played['Games_Total']
    games_played.drop(['Diff_W', 'Diff_L', 'Points_scored_W', 'Points_scored_L', 'Points_scored_against_W', 'Points_scored_against_L'], axis = 1, inplace  = True)
    
    return games_played

Mgames_played = build_stats(MAllResults)
Wgames_played = build_stats(WAllResults)
Mgames_played.head()
plt.figure(figsize = (12,4)) 
plt.subplot(121)
# REPRENDRE ICI
Mgames_played.sort_values('Winning_Rate')['Winning_Rate'].tail(10).plot(kind='barh', title = 'Men Teams with the highest % of win', cmap = plt.get_cmap('tab20c'))
plt.subplot(122)
Wgames_played.sort_values('Winning_Rate')['Winning_Rate'].tail(10).plot(kind='barh', title = 'Women Teams with the highest % of win', style = 'ggplot')
plt.tight_layout()
plt.figure(figsize = (12,4))
plt.subplot(121)
Mgames_played.sort_values('Games_Total')['Games_Total'].tail(10).plot(kind='barh', title = 'Men Teams with most games played')
plt.subplot(122)
Wgames_played.sort_values('Games_Total')['Games_Total'].tail(10).plot(kind='barh', title = 'Women Teams with most games played')
plt.tight_layout()
plt.figure(figsize = (12,8))
plt.subplot(221)
Mgames_played.sort_values('Avg_Diff')['Avg_Diff'].tail(10).plot(kind='barh', title = 'Men Teams with the highest Avg Score Difference')
plt.subplot(222)
Wgames_played.sort_values('Avg_Diff')['Avg_Diff'].tail(10).plot(kind='barh', title = 'Women Teams with the highest Avg Score Difference')
plt.subplot(223)
Mgames_played.sort_values('Avg_Diff')['Avg_Diff'].head(10).plot(kind='barh', title = 'Men Teams with the lowest Avg Score Difference')
plt.subplot(224)
Wgames_played.sort_values('Avg_Diff')['Avg_Diff'].head(10).plot(kind='barh', title = 'Women Teams with the lowest Avg Score Difference')

plt.tight_layout()
plt.figure(figsize=(12,8))
plt.subplot(221)
Mgames_played.sort_values('Avg_Points_scored')['Avg_Points_scored'].tail(10).plot(kind='barh', title = 'Most offensive Men teams')
plt.subplot(222)
Wgames_played.sort_values('Avg_Points_scored')['Avg_Points_scored'].tail(10).plot(kind='barh', title = 'Most offensive Women teams')
plt.subplot(223)
Mgames_played.sort_values('Avg_Points_scored_against')['Avg_Points_scored_against'].tail(10).plot(kind='barh', title = 'Most defensive Men teams')
plt.subplot(224)
Wgames_played.sort_values('Avg_Points_scored_against')['Avg_Points_scored_against'].head(10).plot(kind='barh', title = 'Most defensive Women teams')
plt.tight_layout()
mens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    mens_events.append(pd.read_csv(f'{MENS_DIR}/MEvents{year}.csv'))
MEvents = pd.concat(mens_events)
print(MEvents.shape)
MEvents.head()
womens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    womens_events.append(pd.read_csv(f'{WOMENS_DIR}/WEvents{year}.csv'))
WEvents = pd.concat(womens_events)
print(WEvents.shape)
WEvents.head()
del mens_events
del womens_events
gc.collect()

MEvents = MEvents.merge(MTeams[['TeamName', 'TeamID']],
               left_on='EventTeamID',
               right_on='TeamID',
               validate='many_to_one')
WEvents = WEvents.merge(WTeams[['TeamName', 'TeamID']],
               left_on='EventTeamID',
               right_on='TeamID',
               validate='many_to_one')
def add_events(df, games_played):
    df['counter'] = 1
    for ev in df['EventType'].unique():
        games_played['{}_Count'.format(ev)] = df[df['EventType'] == ev].groupby('TeamName')['counter'].count()
        games_played['{}_Avg'.format(ev)] = games_played['{}_Count'.format(ev)] / games_played['Games_Total_2015']
    games_played.head()
    games_played['Fair Play'] = (games_played['foul_Avg'] - games_played['foul_Avg'].min()) / (games_played['foul_Avg'].max() - games_played['foul_Avg'].min())
    games_played['acc'] = (games_played['made1_Count'] + games_played['made2_Count'] + games_played['made3_Count']) / (games_played['made1_Count'] + games_played['made2_Count'] + games_played['made3_Count'] + games_played['miss1_Count'] + games_played['miss2_Count'] + games_played['miss3_Count'])
    games_played['Accuracy'] = (games_played['acc'] - games_played['acc'].min()) / (games_played['acc'].max() - games_played['acc'].min())
    games_played['Far-shooter'] = (games_played['made3_Avg'] - games_played['made3_Avg'].min()) / (games_played['made3_Avg'].max() - games_played['made3_Avg'].min())
    games_played['Block'] = (games_played['block_Avg'] - games_played['block_Avg'].min()) / (games_played['block_Avg'].max() - games_played['block_Avg'].min())
    games_played['Steal'] = (games_played['steal_Avg'] - games_played['steal_Avg'].min()) / (games_played['steal_Avg'].max() - games_played['steal_Avg'].min())
    
    return games_played

Mgames_played = add_events(MEvents, Mgames_played)
Wgames_played = add_events(WEvents, Wgames_played)
#Code adapted from https://typewind.github.io/2017/09/29/radar-chart/
labels = ['Fair Play', 'Accuracy', 'Far-shooter', 'Block', 'Steal']
plt.figure(figsize = (12,4))

def build_radar(team, num_subplot):
    stats = Mgames_played.loc[team][labels].values
    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False) # Set the angle
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    fig=plt.figure()
    ax = fig.add_subplot(num_subplot, polar=True)   # Set polar axis
    ax.plot(angles, stats, 'o-', linewidth=2)  # Draw the plot (or the frame on the radar chart)
    ax.fill(angles, stats, alpha=0.25)  #Fulfill the area
    ax.set_thetagrids(angles * 180/np.pi, labels)  # Set the label for each axis
    ax.set_title(Mgames_played.loc[team].name) 
    ax.set_rlim(0,1)
    ax.grid(True)


build_radar('Duke', 121)
build_radar('Longwood', 122)
MMassey = pd.read_csv(f'{MENS_DIR}/MDataFiles_Stage1/MMasseyOrdinals.csv')
MMassey = MMassey.merge(MTeams[['TeamName', 'TeamID']],
               left_on='TeamID',
               right_on='TeamID',
               validate='many_to_one')


best_teams = MMassey[(MMassey['SystemName'] == 'MAS') 
                      & (MMassey['OrdinalRank'] <= 3) 
                      & (MMassey['Season'] == 2019)]['TeamName'].unique()
rank_df = pd.DataFrame(columns = ['TeamName', 'OrdinalRank', 'Season', 'Temp'])
for season in MMassey['Season'].unique():
    k = 0
    counter = 0
    for i in np.linspace(0, 133, 3, dtype = int):
        rank_series = pd.DataFrame(MMassey[(MMassey['SystemName'] == 'MAS')
                      & (MMassey['Season'] == season)
                      & (MMassey['RankingDayNum'] <= i) 
                      & (MMassey['RankingDayNum'] >= k)
                      & (MMassey['TeamName'].isin(best_teams))].groupby('TeamName', as_index=False)['OrdinalRank'].mean())
        rank_series['Season'] = season
        rank_series['Temp'] = counter
        k = i
        counter += 1
        rank_df = rank_df.append(rank_series)
rank_df.head()
ranking_table = pd.pivot_table(rank_df,
               values = 'OrdinalRank',
               index = 'TeamName',
               columns = ['Season', 'Temp'],
               aggfunc= (lambda x: x))
ranking_table.head()
ranking_table.T.plot(legend = True, 
                     colormap = 'Set1',
                     figsize = (10,10), 
                     title = 'Average ranking of last years\' best teams',
                     linewidth = 1)
plt.gca().invert_yaxis()
plt.ylim(bottom=100, top = 0)
plt.show()
