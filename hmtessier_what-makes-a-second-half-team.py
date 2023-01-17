#imports

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import statsmodels.api as sm

from scipy.stats import ttest_rel, ttest_ind, chi2_contingency

from scipy import stats

pd.options.display.max_rows = 500

plt.style.use('fivethirtyeight')
#create TeamID to name dictionary

team_names = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv')

team_names = team_names[['TeamName','TeamID']]

team_dict = dict(zip(team_names.TeamID, team_names.TeamName))
def clean_plays(playframe, team_dict):

    """Clean play by play dataframe for entire season"""

    

    df = playframe.copy()

    

    #map team IDs to team names

    df.WTeamID = df.WTeamID.map(team_dict)

    df.LTeamID = df.LTeamID.map(team_dict)

    df.EventTeamID = df.EventTeamID.map(team_dict)

    

    #filter unnecessray columns

    df = df[['EventID','Season','DayNum','WTeamID','LTeamID','WCurrentScore','LCurrentScore','ElapsedSeconds','EventType']]

    

    #remove all non-scoring event types

    df = df[df.EventType.isin(['made1','made2','made3'])]

    

    #create unique game IDs

    df['GameID'] = df.groupby(['Season','DayNum','WTeamID','LTeamID']).grouper.group_info[0]

    

    #change buzzer-beater scores ElapsedSeconds to be slightly before end of time (for correct game minute calc)

    df.loc[(df.ElapsedSeconds == 1200), 'ElapsedSeconds'] = 1199.99 #Halftime

    df.loc[(df.ElapsedSeconds == 2400), 'ElapsedSeconds'] = 2399.99 #End Regulation

    df.loc[(df.ElapsedSeconds == 2700), 'ElapsedSeconds'] = 2699.99 #End OT1

    df.loc[(df.ElapsedSeconds == 3000), 'ElapsedSeconds'] = 2999.99 #End OT2

    df.loc[(df.ElapsedSeconds == 3300), 'ElapsedSeconds'] = 3299.99 #End OT3

    df.loc[(df.ElapsedSeconds == 3700), 'ElapsedSeconds'] = 3699.99 #End OT4

    

    #calculate game period

    df['Period'] = df['ElapsedSeconds'] // 150



    return df



#read in 2020 data

plays20 = pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2020.csv')

clean_M20 = clean_plays(plays20, team_dict)
plays20.EventType.value_counts().plot(kind='bar')

plt.title('2019-2020 Season Event Counts by Type')

plt.show()
def get_game(clean_plays_df, gameID, teamName):

    """Given a team and game, return point differentials for each period"""

    

    temp = clean_plays_df[clean_plays_df.GameID == gameID].sort_values('ElapsedSeconds')

    temp = temp[['WTeamID','LTeamID','WCurrentScore','LCurrentScore','Period']]

    

    #find last minute recorded in data (determine number of OTs)

    upper = max(temp.Period)

    #reindex to fill in missing minutes where no team scored depending on number of OTs

    if upper < 16:

        #get last score for each minute

        temp = temp.groupby('Period').last().reindex(index=range(0,16), method = 'ffill')

    elif upper < 18:

        temp = temp.groupby('Period').last().reindex(index=range(0,18), method = 'ffill')

    elif upper < 20:

        temp = temp.groupby('Period').last().reindex(index=range(0,20), method = 'ffill')

    elif upper < 22:

        temp = temp.groupby('Period').last().reindex(index=range(0,22), method = 'ffill')

    elif upper < 24:

        temp = temp.groupby('Period').last().reindex(index=range(0,24), method = 'ffill')

    elif upper < 26:

        temp = temp.groupby('Period').last().reindex(index=range(0,26), method = 'ffill')

    

    #fill missing values

    temp[['WTeamID','LTeamID']] = temp[['WTeamID','LTeamID']].bfill()

    temp[['WCurrentScore','LCurrentScore']] = temp[['WCurrentScore','LCurrentScore']].fillna(0)

    

    #fix data issue where LCurrentScore and WCurrentScore values are swapped

    winning_final = temp.WCurrentScore.iloc[-1]

    losing_final = temp.LCurrentScore.iloc[-1]



    if winning_final < losing_final:

        temp.loc[a.index[-1], 'WCurrentScore'] = losing_final

        temp.loc[a.index[-1], 'LCurrentScore'] = winning_final

    

    #calc period point differentials for each team

    temp['Diff'] = temp.WCurrentScore - temp.LCurrentScore

    temp['WScoreDiff'] = temp['Diff'].diff()

    temp['LScoreDiff'] = temp['Diff'].diff() * -1

    

    #point diffs for first period

    min1diff = temp['WCurrentScore'][0] - temp['LCurrentScore'][0]

    temp.loc[0, 'WScoreDiff'] = min1diff

    temp.loc[0,'LScoreDiff'] = min1diff * -1

    

    #return column for team we're interested in

    if teamName == temp.loc[0, 'WTeamID']:

        rel =  temp['WScoreDiff']

    elif teamName == temp.loc[0, 'LTeamID']:

        rel =  temp['LScoreDiff']

    

    return rel 



def find_team_games(team, playsframe):

    """find all games in which a team played during the season"""

    return playsframe[(playsframe.WTeamID == team) | (playsframe.LTeamID == team)].GameID.unique()



def generate_series(Team_Name, play_frame):

    """Given a team and clean play by play dataframe return differentials dataframe"""

    

    games = find_team_games(Team_Name, play_frame)

    diff_list = []

    for GameID in games:

        diff_list += [list(get_game(play_frame, GameID, Team_Name))]

    diff_frame = pd.DataFrame(diff_list).transpose()



    differentials = pd.DataFrame({'Differential':diff_frame.mean(axis=1)})

    differentials = differentials - differentials[:16].mean()

    

    game_counts = pd.DataFrame({'Games':diff_frame.count(axis = 1)})

    return differentials.join(game_counts)



def resilience_score(Team_Name, play_frame):

    """Given a team and clean play by play dataframe calculate team's resileince score for season"""

    

    resil_df = generate_series(Team_Name, play_frame)

    resil_df['GameWeight'] = resil_df['Games'] / max(resil_df['Games']) #adjust weight of OTs based on num games played

    resil_df['WeightedDifferential'] = resil_df['GameWeight'] * resil_df['Differential'] #recalc weighted point differentials

    resil = stats.linregress(y = resil_df['WeightedDifferential'], x = resil_df.index)[0] #return slope of linear regression

    return resil * 100

    

def plot_team(Team_Name, play_frame):

    """Given a team and clean play by play dataframe plot point standardized point differentials"""

    

    df = generate_series(Team_Name, play_frame)

    df['GameWeight'] = df['Games'] / max(df['Games'])

    df['WeightedDifferential'] = df['GameWeight'] * df['Differential']

    pal = sns.color_palette('RdBu_r', len(df))

    rank = df.Differential.argsort().argsort()

    g = sns.barplot(y = 'WeightedDifferential', x = df.index, data = df, palette = np.array(pal[::-1])[rank])

    g = sns.regplot(y = 'WeightedDifferential', x = df.index, data = df, scatter = False, ci=None, line_kws={'color':'g', 'lw':3})

    plt.title(Team_Name)

    plt.ylabel('Point Differential')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    plt.axvline(7.5, color = 'green', ls = '--', linewidth = 1)

    bottom = min(df.WeightedDifferential) - (max(df.WeightedDifferential) - min(df.WeightedDifferential))/10

    plt.text(2, bottom , '1st Half')

    plt.text(10, bottom, '2nd Half')

    if len(df) > 16:

        plt.axvline(15.5, color = 'green', ls = '--', linewidth = 1)

        if len(df) > 18:

            plt.text(17, bottom, 'OT')

        else:

            plt.text(16, bottom, 'OT')  

    plt.show()
def complete_season_data(team_dict, play_frame):

    teams = []

    resilience = []

    for team in team_dict.values():

        try:

            resilience += [resilience_score(team, play_frame)]

            teams += [team]

        except: #not all teams were D1 for all seasons in data

            pass

    resiliences = pd.DataFrame({'Team':teams, 'Resilience':resilience})

    resiliences = resiliences.sort_values('Resilience', ascending=False).reset_index(drop=True)

    return resiliences



resilience_df = complete_season_data(team_dict, clean_M20)
kenpom20 = pd.read_csv('../input/kpom2020/kpom20.csv')

merged = kenpom20.merge(resilience_df)
fig, ax = plt.subplots(1,2, figsize=(12,6))

sns.regplot(x = 'AdjEM', y = 'Resilience', data = merged, ci=None, line_kws={'color':'red', 'lw':2}, ax = ax[0])

sns.regplot(x = 'SOS', y = 'Resilience', data = merged, ci=None, line_kws={'color':'red', 'lw':2}, ax = ax[1])

fig.suptitle('Skill and Strength of Schedule vs Resilience')

fig.show()
#change functions to remove blowout games

def get_game(clean_plays_df, gameID, teamName):

    """Given a team and game, return point differentials for each period"""

    

    temp = clean_plays_df[clean_plays_df.GameID == gameID].sort_values('ElapsedSeconds')

    temp = temp[['WTeamID','LTeamID','WCurrentScore','LCurrentScore','Period']]

    

    #find last minute recorded in data (determine number of OTs)

    upper = max(temp.Period)

    #reindex to fill in missing minutes where no team scored depending on number of OTs

    if upper < 16:

        #get last score for each minute

        temp = temp.groupby('Period').last().reindex(index=range(0,16), method = 'ffill')

    elif upper < 18:

        temp = temp.groupby('Period').last().reindex(index=range(0,18), method = 'ffill')

    elif upper < 20:

        temp = temp.groupby('Period').last().reindex(index=range(0,20), method = 'ffill')

    elif upper < 22:

        temp = temp.groupby('Period').last().reindex(index=range(0,22), method = 'ffill')

    elif upper < 24:

        temp = temp.groupby('Period').last().reindex(index=range(0,24), method = 'ffill')

    elif upper < 26:

        temp = temp.groupby('Period').last().reindex(index=range(0,26), method = 'ffill')

    

    #fill missing values

    temp[['WTeamID','LTeamID']] = temp[['WTeamID','LTeamID']].bfill()

    temp[['WCurrentScore','LCurrentScore']] = temp[['WCurrentScore','LCurrentScore']].fillna(0)

    

    #fix data issue where LCurrentScore and WCurrentScore values are swapped

    winning_final = temp.WCurrentScore.iloc[-1]

    losing_final = temp.LCurrentScore.iloc[-1]

    if winning_final < losing_final:

        temp.loc[a.index[-1], 'WCurrentScore'] = losing_final

        temp.loc[a.index[-1], 'LCurrentScore'] = winning_final

    

    #calc period point differentials for each team

    temp['Diff'] = temp.WCurrentScore - temp.LCurrentScore

    temp['WScoreDiff'] = temp['Diff'].diff()

    temp['LScoreDiff'] = temp['Diff'].diff() * -1

    

    #point diffs for first period

    min1diff = temp['WCurrentScore'][0] - temp['LCurrentScore'][0]

    temp.loc[0, 'WScoreDiff'] = min1diff

    temp.loc[0,'LScoreDiff'] = min1diff * -1



    #calculate blowout index 

    temp['BlowoutScore'] = np.sqrt(2400 - 150*(temp.reset_index().Period.iloc[:15] + 1)) + 2.5 < temp.Diff.iloc[:15]

    if temp['BlowoutScore'].iloc[:15].any():

        return None

    

    #return column for team we're interested in

    if teamName == temp.loc[0, 'WTeamID']:

        rel =  temp['WScoreDiff']

    elif teamName == temp.loc[0, 'LTeamID']:

        rel =  temp['LScoreDiff']

        

    return rel



def generate_series(Team_Name, play_frame):

    """Given a team and clean play by play dataframe return differentials dataframe"""

    

    games = find_team_games(Team_Name, play_frame)

    diff_list = []

    for GameID in games:

        try:

            diff_list += [list(get_game(play_frame, GameID, Team_Name))]

        except:

            pass

    diff_frame = pd.DataFrame(diff_list).transpose()



    differentials = pd.DataFrame({'Differential':diff_frame.mean(axis=1)})

    differentials = differentials - differentials[:16].mean()

    

    game_counts = pd.DataFrame({'Games':diff_frame.count(axis = 1)})

    return differentials.join(game_counts)
resilience_df = complete_season_data(team_dict, clean_M20)

merged = kenpom20.merge(resilience_df)
fig, ax =plt.subplots(1,2, figsize=(12,6))

sns.regplot(x = 'AdjEM', y = 'Resilience', data = merged, ci=None, line_kws={'color':'red', 'lw':2}, ax = ax[0])

sns.regplot(x = 'SOS', y = 'Resilience', data = merged, ci=None, line_kws={'color':'red', 'lw':2}, ax = ax[1])

fig.suptitle('Skill and Strength of Schedule vs Resilience (Blowout Adjusted)')

fig.show()
model = sm.OLS(merged.Resilience, merged.AdjEM).fit()

model.summary()
plot_team('Virginia', clean_M20)
resilience_df[resilience_df.Team == 'Virginia']
resilience_df.head()
plot_team('Maryland', clean_M20)
resilience_df.tail()
plot_team('Penn St',clean_M20)
resilience_df.style.background_gradient(cmap='RdBu')
resilience_df.to_csv('M20.csv', index=False)
#read in historical data, remove tournament data, calculate resilience scores

clean_M19 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2019.csv'), team_dict)

clean_M19 = clean_M19[clean_M19.DayNum < 133]

resilience_19 = complete_season_data(team_dict, clean_M19)



clean_M18 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2018.csv'), team_dict)

clean_M18 = clean_M18[clean_M18.DayNum < 133]

resilience_18 = complete_season_data(team_dict, clean_M18)



clean_M17 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2017.csv'), team_dict)

clean_M17 = clean_M17[clean_M17.DayNum < 133]

resilience_17 = complete_season_data(team_dict, clean_M17)



clean_M16 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2016.csv'), team_dict)

clean_M16 = clean_M16[clean_M16.DayNum < 133]

resilience_16 = complete_season_data(team_dict, clean_M16)



clean_M15 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2015.csv'), team_dict)

clean_M15 = clean_M15[clean_M15.DayNum < 133]

resilience_15 = complete_season_data(team_dict, clean_M15)



#Add year column prior to merge

resilience_19['Year'] = '2019'

resilience_18['Year'] = '2018'

resilience_17['Year'] = '2017'

resilience_16['Year'] = '2016'

resilience_15['Year'] = '2015'



#merge

historical_resilience = pd.concat([resilience_19, resilience_18, resilience_17, resilience_16, resilience_15], ignore_index=True)

historical_resilience.to_csv('HistoricalResilience.csv', index=False)
seeds = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneySeeds.csv')

seeds = seeds[seeds.Season >= 2015].reset_index(drop=True)

seeds['Team'] = seeds['TeamID'].map(team_dict)

seeds['Seed'] = seeds['Seed'].apply(lambda x: ''.join([digit for digit in x if digit.isdigit()])).astype(int)

seeds = seeds[['Season','Seed','Team']]

seeds.columns = ['Year','Seed','Team']



#match resilience dtypes

historical_resilience['Year'] = historical_resilience['Year'].astype('int')

historical_resilience['Team'] = historical_resilience['Team'].astype('str')



#merge dataframes

res_seed = seeds.merge(historical_resilience, left_on = ['Team','Year'], right_on = ['Team','Year'])



sns.regplot(x = 'Seed', y = 'Resilience', data = res_seed, ci=None, line_kws={'color':'red', 'lw':2})

plt.title('Resilience vs NCAA Tournamet Seed (Since 2015)')

plt.show()
spreads = pd.read_csv('../input/tournamentpointspreads/Spreads.csv')



spreads['WResilience'] *= 100

spreads['LResilience'] *= 100



resilience = []

performance = []

for index, row in spreads.iterrows():

    resilience += [row.WResilience]

    performance += [row.Wspread - row.TrueSpread]

    resilience += [row.LResilience]

    performance += [(row.Wspread - row.TrueSpread)*-1]

    

spread_performance = pd.DataFrame({'Resilience':resilience, 'Performance':performance})



sns.regplot(x = 'Resilience', y = 'Performance', data = spread_performance, ci=None, line_kws={'color':'red', 'lw':2})

plt.title('Performance (Against Point Spread)\nvs Resilience')

plt.show()
#read in and clean tournament game results data

results = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')

results['WTeamID'] = results['WTeamID'].map(team_dict)

results['LTeamID'] = results['LTeamID'].map(team_dict)

results = results[results.Season >= 2015].reset_index()

results = results[['Season','WTeamID','LTeamID','WScore','LScore']]



results = results.merge(res_seed, left_on = ['WTeamID','Season'], right_on = ['Team','Year'])

results = results[['Year','WTeamID','LTeamID','WScore','LScore','Seed','Resilience']]

results.columns = ['Year','WTeam','LTeam','WScore','LScore','WSeed','WResilience']

results = results.merge(res_seed, left_on = ['Year','LTeam'], right_on = ['Year','Team'])

results = results[['Year','WTeam','LTeam','WSeed','Seed','WResilience','Resilience','WScore','LScore']]

results.columns = ['Year','WTeam','LTeam','WSeed','LSeed','WResilience','LResilience','WScore','LScore']



results.to_csv('Results.csv', index=False)



print('Average winning team resilience: ' + str(round(results.WResilience.mean(), 4)))

print('Average losing team resilience: ' + str(round(results.LResilience.mean(), 4)))
ttest_rel(results.WResilience, results.LResilience)
reg_sn = pd.read_csv('../input/regularseasonpointspreads/PointSpreads2015-20RegularSeason.csv')

reg_sn['Result'] = reg_sn.hscore - reg_sn.rscore

reg_sn['Upset'] = (reg_sn.Result * reg_sn.line) < 0 

print('NCAA Men\'s Baskeball Upset Ratio 2015-2020:\n')

print('NCAA Tournament: ' + str(round(sum(spreads['Wspread'] > 0)/len(spreads),4)))

print('Regular Season: ' + str(round(sum(reg_sn.Upset)/len(reg_sn),4)))
clean_M19 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2019.csv'), team_dict)

clean_M19 = clean_M19[clean_M19.DayNum >= 133]



clean_M18 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2018.csv'), team_dict)

clean_M18 = clean_M18[clean_M18.DayNum >= 133]



clean_M17 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2017.csv'), team_dict)

clean_M17 = clean_M17[clean_M17.DayNum >= 133]



clean_M16 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2016.csv'), team_dict)

clean_M16 = clean_M16[clean_M16.DayNum >= 133]



clean_M15 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2015.csv'), team_dict)

clean_M15 = clean_M15[clean_M15.DayNum >= 133]



tourney_pbp = pd.concat([clean_M19, clean_M18, clean_M17, clean_M16, clean_M15], ignore_index=True)

tourney_pbp['GameID'] = tourney_pbp.groupby(['Season','WTeamID','LTeamID']).grouper.group_info[0]



WTeam = []

LTeam = []

comeback_amt = []

season = []

for gameid in tourney_pbp.GameID.unique():

    game_frame = tourney_pbp[tourney_pbp.GameID == gameid]

    diff = game_frame.WCurrentScore - game_frame.LCurrentScore

    comeback_amt += [min(diff)]

    WTeam += [game_frame.WTeamID.iloc[0]]

    LTeam += [game_frame.LTeamID.iloc[0]]

    season += [game_frame.Season.iloc[0]]

    

comebacks = pd.DataFrame({'Season':season,'WTeam':WTeam,'LTeam':LTeam,'Comeback_Amt':comeback_amt}).merge(spreads, left_on = ['Season','WTeam','LTeam'], right_on = ['Year','WTeam','LTeam'])

comebacks['ResilienceDifference'] = comebacks.WResilience - comebacks.LResilience

comebacks['Comeback_Amt'] = comebacks.Comeback_Amt * -1



grouped = comebacks[(comebacks.Comeback_Amt > 0) & (comebacks.Comeback_Amt < 20)].groupby('Comeback_Amt')['WResilience','LResilience','ResilienceDifference'].mean()



pal = sns.color_palette('RdBu_r', len(grouped))

rank = grouped.ResilienceDifference.argsort().argsort()

sns.barplot(grouped.index,'ResilienceDifference', data = grouped, palette = np.array(pal[::-1])[rank])

plt.xlabel('Comeback Amount (Points)')

plt.ylabel('Average Resilience Difference')

plt.title('Average Difference in Resilience\n(Winning Team - Losing Team)\nby Size of Comeback')

plt.show()
WTeam = []

LTeam = []

season = []

halftime = []

for gameid in tourney_pbp.GameID.unique():

    game_frame = tourney_pbp[tourney_pbp.GameID == gameid].groupby('Period').last()

    diff = game_frame.WCurrentScore - game_frame.LCurrentScore

    try:

        ht = diff[7]

    except:

        try:

            ht = diff[8]

        except:

            ht = 0

    if abs(ht) > 0:

        halftime += [ht]

        WTeam += [game_frame.WTeamID.iloc[0]]

        LTeam += [game_frame.LTeamID.iloc[0]]

        season += [game_frame.Season.iloc[0]]



res_mtx = pd.DataFrame({'Season':season,'WTeam':WTeam,'LTeam':LTeam,'Halftime':halftime}).merge(spreads, left_on = ['Season','WTeam','LTeam'], right_on = ['Year','WTeam','LTeam'])



team = []

halftime = []

resilience_diff = []

result = []

for index, row in res_mtx.iterrows():

    team += [row.WTeam]

    halftime += [row.Halftime]

    resilience_diff += [row.WResilience - row.LResilience]

    result += ['Win']

    

    team += [row.LTeam]

    halftime += [row.Halftime *-1]

    resilience_diff += [row.LResilience - row.WResilience]

    result += ['Loss']

    

matrix = pd.DataFrame({'Team':team,'Halftime':halftime,'Resilience_Diff':resilience_diff,'Result':result})



#win pct when winning at halftime by more than 5 and higher resilience

temp = matrix[(matrix.Halftime > 0) & (matrix.Resilience_Diff > 0)]

higher_winning = len(temp[temp.Result == 'Win'])/len(temp)



#win pct when losing at halftime by more than 5 and higher resilience

temp = matrix[(matrix.Halftime < 0) & (matrix.Resilience_Diff > 0)]

higher_losing = len(temp[temp.Result == 'Win'])/len(temp)



#win pct when winning at halftime by more than 5 and lower resilience

temp = matrix[(matrix.Halftime > 0) & (matrix.Resilience_Diff < 0)]

lower_winning = len(temp[temp.Result == 'Win'])/len(temp)



#win pct when losing at halftime by more than 5 and lower resilience

temp = matrix[(matrix.Halftime < 0) & (matrix.Resilience_Diff < 0)]

lower_losing = len(temp[temp.Result == 'Win'])/len(temp)



win_pct = np.array([[higher_winning, higher_losing],[lower_winning, lower_losing]])

sns.heatmap(win_pct, center = .50, cmap='coolwarm_r', annot=True, fmt=".1%", cbar=False)

plt.xticks([0.5,1.5],['Winning','Losing'])

plt.yticks([0.3,1.3],['Higher','Lower'])

plt.xlabel('Halftime Score')

plt.ylabel('Resilience\n(Compared to Opponent)')

plt.title('NCAA Tournament Game Win Percentage by\nHalftime Score and Resilience')

plt.show()
clean_M19 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2019.csv'), team_dict)

clean_M18 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2018.csv'), team_dict)

clean_M17 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2017.csv'), team_dict)

clean_M16 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2016.csv'), team_dict)

clean_M15 = clean_plays(pd.read_csv('../input/march-madness-analytics-2020/MPlayByPlay_Stage2/MEvents2015.csv'), team_dict)



all_pbp = pd.concat([clean_M19, clean_M18, clean_M17, clean_M16, clean_M15], ignore_index=True)

all_pbp['GameID'] = all_pbp.groupby(['Season','WTeamID','LTeamID']).grouper.group_info[0]



WTeam = []

LTeam = []

season = []

halftime = []

for gameid in all_pbp.GameID.unique():

    game_frame = all_pbp[all_pbp.GameID == gameid].groupby('Period').last()

    diff = game_frame.WCurrentScore - game_frame.LCurrentScore

    try:

        ht = diff[7]

    except:

        try:

            ht = diff[8]

        except:

            ht = 0

    if abs(ht) > 0:

        halftime += [ht]

        WTeam += [game_frame.WTeamID.iloc[0]]

        LTeam += [game_frame.LTeamID.iloc[0]]

        season += [game_frame.Season.iloc[0]]



res_mtx = pd.DataFrame({'Season':season,'WTeam':WTeam,'LTeam':LTeam,'Halftime':halftime})



res_mtx = res_mtx.merge(historical_resilience, left_on = ['Season','WTeam'], right_on = ['Year','Team'])

res_mtx = res_mtx[['Season','WTeam','LTeam','Halftime','Resilience']]

res_mtx.columns = ['Season','WTeam','LTeam','Halftime','WResilience']

res_mtx = res_mtx.merge(historical_resilience, left_on = ['Season','LTeam'], right_on = ['Year','Team'])

res_mtx = res_mtx[['Season','WTeam','LTeam','Halftime','WResilience','Resilience']]

res_mtx.columns = ['Season','WTeam','LTeam','Halftime','WResilience','LResilience']



team = []

halftime = []

resilience_diff = []

result = []

for index, row in res_mtx.iterrows():

    team += [row.WTeam]

    halftime += [row.Halftime]

    resilience_diff += [row.WResilience - row.LResilience]

    result += ['Win']

    

    team += [row.LTeam]

    halftime += [row.Halftime *-1]

    resilience_diff += [row.LResilience - row.WResilience]

    result += ['Loss']

    

matrix = pd.DataFrame({'Team':team,'Halftime':halftime,'Resilience_Diff':resilience_diff,'Result':result})





#contingency matrix generated by recording counts from matrix dataframe 

chi2_contingency(np.array([[7984, 1863],[8858,2996]]))
WTeam = []

LTeam = []

season = []

halftime = []

for gameid in tourney_pbp.GameID.unique():

    game_frame = tourney_pbp[tourney_pbp.GameID == gameid].groupby('Period').last()

    diff = game_frame.WCurrentScore - game_frame.LCurrentScore

    try:

        ht = diff[7]

    except:

        try:

            ht = diff[8]

        except:

            ht = 0

    if abs(ht) > 5: #only games where halftime differential is greater than 5.

        halftime += [ht]

        WTeam += [game_frame.WTeamID.iloc[0]]

        LTeam += [game_frame.LTeamID.iloc[0]]

        season += [game_frame.Season.iloc[0]]



res_mtx = pd.DataFrame({'Season':season,'WTeam':WTeam,'LTeam':LTeam,'Halftime':halftime}).merge(spreads, left_on = ['Season','WTeam','LTeam'], right_on = ['Year','WTeam','LTeam'])



team = []

halftime = []

resilience_diff = []

result = []

for index, row in res_mtx.iterrows():

    team += [row.WTeam]

    halftime += [row.Halftime]

    resilience_diff += [row.WResilience - row.LResilience]

    result += ['Win']

    

    team += [row.LTeam]

    halftime += [row.Halftime *-1]

    resilience_diff += [row.LResilience - row.WResilience]

    result += ['Loss']

    

matrix = pd.DataFrame({'Team':team,'Halftime':halftime,'Resilience_Diff':resilience_diff,'Result':result})



#win pct when winning at halftime by more than 5 and higher resilience

temp = matrix[(matrix.Halftime > 0) & (matrix.Resilience_Diff > 0)]

higher_winning = len(temp[temp.Result == 'Win'])/len(temp)



#win pct when losing at halftime by more than 5 and higher resilience

temp = matrix[(matrix.Halftime < 0) & (matrix.Resilience_Diff > 0)]

higher_losing = len(temp[temp.Result == 'Win'])/len(temp)



#win pct when winning at halftime by more than 5 and lower resilience

temp = matrix[(matrix.Halftime > 0) & (matrix.Resilience_Diff < 0)]

lower_winning = len(temp[temp.Result == 'Win'])/len(temp)



#win pct when losing at halftime by more than 5 and lower resilience

temp = matrix[(matrix.Halftime < 0) & (matrix.Resilience_Diff < 0)]

lower_losing = len(temp[temp.Result == 'Win'])/len(temp)



win_pct = np.array([[higher_winning, higher_losing],[lower_winning, lower_losing]])

sns.heatmap(win_pct, center = .50, cmap='coolwarm_r', annot=True, fmt=".1%", cbar=False)

plt.xticks([0.5,1.5],['Winning by\n> 5 Points','Losing by\n> 5 Points'])

plt.yticks([0.3,1.3],['Higher','Lower'])

plt.xlabel('Halftime Score')

plt.ylabel('Resilience\n(Compared to Opponent)')

plt.title('NCAA Tournament Game Win Percentage by\nHalftime Score and Resilience')

plt.show()
a = matrix[matrix.Resilience_Diff > 0]

print('Overall Higher Resilience Win Percentage: ' + str(round(len(a[a.Result == 'Win'])/len(a) * 100, 2)) + '%')