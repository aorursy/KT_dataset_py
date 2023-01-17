import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
dir_prefix = '/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'
teams = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MTeams.csv")

seasons = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MSeasons.csv")

tourney_seeds = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MNCAATourneySeeds.csv")

reg_season_result = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MRegularSeasonCompactResults.csv")

tournament_result = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
reg_season_gbg = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MRegularSeasonDetailedResults.csv")

tournament_gbg = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MNCAATourneyDetailedResults.csv")
cities = pd.read_csv(dir_prefix + "MDataFiles_Stage1/Cities.csv")

game_cities = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MGameCities.csv")
rankings = pd.read_csv(dir_prefix + "MDataFiles_Stage1/MMasseyOrdinals.csv")
"""

Calculate offensive and defensive efficiency of a team throughout the season

"""

def calcEfficiency(dataset,teamID):

    game_won = dataset.loc[dataset['WTeamID'] == teamID]

    game_lost = dataset.loc[dataset['LTeamID'] == teamID]

    total_score = 0

    total_score_allowed = 0

    total_FGA = 0

    total_OR = 0

    total_TO = 0

    total_FTA = 0

    for index, row in game_won.iterrows():

        total_score = total_score + row['WScore']

        total_score_allowed = total_score_allowed + row['LScore']

        total_FGA = total_FGA + row['WFGA']

        total_OR = total_OR + row['WOR']

        total_TO = total_TO + row['WTO']

        total_FTA = total_FTA + row['WFTA']

    for index, row in game_lost.iterrows():

        total_score = total_score + row['LScore']

        total_score_allowed = total_score_allowed + row['WScore']

        total_FGA = total_FGA + row['LFGA']

        total_OR = total_OR + row['LOR']

        total_TO = total_TO + row['LTO']

        total_FTA = total_FTA + row['LFTA']

    poss = total_FGA - total_OR + total_TO + 0.4 * total_FTA

    if teamID != 1309:

        oeff = total_score / poss

        deff = total_score_allowed / poss

        return oeff, deff

    else:

        return 0,0
"""

Calculate efficiency for all teams in a single season

"""

def applyEff_team(dataset, year):

    OEff = []

    DEff = []

    for team, lastYear, firstYear in zip(teams['TeamID'],teams['LastD1Season'], teams['FirstD1Season']):

        if (firstYear <= year) and (lastYear >= year):

            OEff_score, DEff_score = calcEfficiency(dataset,team)

            OEff.append(OEff_score)

            DEff.append(DEff_score)

        else:

            OEff.append('NaN')

            DEff.append('NaN')

    return OEff, DEff
"""

This function create ID and label for each game by default

Optional parameters to calculate Seed_diff and ranking_diff is available

"""

def featureAddition(dataset,seed=None, rank=None, elo=None):

    for index, row in dataset.iterrows():

        lower_ID = row['WTeamID']

        higher_ID = row['LTeamID']

        if seed != None:

            seed_diff = int(row['WTeamSeedPure']) - int(row['LTeamSeedPure'])

        if rank != None:

            ranking_diff = int(row['WRanking']) - int(row['LRanking'])

        if elo != None:

            lower_elo = row['w_elo']

            higher_elo = row['l_elo']



        if lower_ID > higher_ID:

            tmp = lower_ID

            lower_ID = higher_ID

            higher_ID = tmp

            if elo != None:

                tmp_elo = lower_elo

                lower_elo = higher_elo

                higher_elo = tmp_elo

            if rank != None:

                ranking_diff = int(row['LRanking']) - int(row['WRanking'])

            if seed != None:

                seed_diff = int(row['LTeamSeedPure']) - int(row['WTeamSeedPure'])

        #Id

        dataset.loc[index, 'ID'] = (str(row['Season']) + "_" + str(lower_ID) + "_" + str(higher_ID))

        #Label

        if lower_ID == row['WTeamID']:

            dataset.loc[index,'lower_win'] = 1

        else:

            dataset.loc[index,'lower_win'] = 0

        #Seed diff

        if seed != None:

            dataset.loc[index, 'Seed_diff'] = seed_diff

        #Ranking diff

        if rank != None:

            dataset.loc[index, 'Ranking_diff'] = ranking_diff

        if elo != None:

            dataset.loc[index, 'lower_elo'] = lower_elo

            dataset.loc[index,'higher_elo'] = higher_elo
"""

Apply efficiency of teams into a single season dataset

"""

def applyEff_season(dataset,year=None):

    for index, row in dataset.iterrows():

        if year == None:

            season = row['Season']

        else:

            season = year

        firstT, secondT = getTeam(dataset, index)

        dataset.loc[index,'Team1_OEff'] = round(float(teams[teams['TeamID'] == int(firstT)]['Offensive_Eff_' + str(season)].values) * 100,1)

        dataset.loc[index,'Team1_DEff'] = round(float(teams[teams['TeamID'] == int(firstT)]['Defensive_Eff_' + str(season)].values) * 100,1)

        dataset.loc[index,'Team2_OEff'] = round(float(teams[teams['TeamID'] == int(secondT)]['Offensive_Eff_' + str(season)].values) * 100,1)

        dataset.loc[index,'Team2_DEff'] = round(float(teams[teams['TeamID'] == int(secondT)]['Defensive_Eff_' + str(season)].values) * 100,1)
"""

Separate seed into separate seasons

"""

def seedSeparation(years):

    tournament_seeds = []

    for year in years:

        tournament_seeds.append(tourney_seeds.groupby(['Season']).get_group(year))

    return tournament_seeds
"""

Get teamID based on gameID

"""

def getTeam(dataset,index):

    season = dataset['ID'][index][0:4]

    firstT = dataset['ID'][index][5:9]

    secondT = dataset['ID'][index][10:14]

    return season, firstT, secondT
def appendPred(df):

    pred = voting_clf.predict_proba(df[['log_lower_elo','log_higher_elo']])

    df['Pred'] = pred[:,1]
def generateGbgSeparation(years):

    reg_gbg = []

    for year in years:

        reg_gbg.append(reg_season_gbg[reg_season_gbg['Season'] == year])

    return reg_gbg
"""

Separate regular season and tournament game results into separate seasons

"""

def yearSeparation(years):

    reg_season_sep = []

    tournament_sep = []

    for year in years:

        reg_season_sep.append(reg_season_result.loc[reg_season_result['Season'] == year])

        tournament_sep.append(tournament_result.loc[tournament_result['Season'] == year])

    #Sort tournament games by DayNum

    for i in range(len(years)):

        tournament_sep[i] = tournament_sep[i].sort_values('DayNum')

    return reg_season_sep, tournament_sep
"""

Bin elo score

"""

def elo_bin(df, attr, bins=3):

    df['tmp'] = pd.cut(df[attr],bins)

    int_len = len(df['tmp'].unique())

    for itv in range(int_len):

        left_bound = (df['tmp'].unique()[int_len - 1 -itv]).left

        right_bound = (df['tmp'].unique()[int_len - 1 - itv]).right

        curr_interval = pd.Interval(left=left_bound, right=right_bound)

        df.loc[(df[attr] == curr_interval), attr] = itv + 1



    df[attr] = df[attr].astype(int)

    df.drop('tmp', axis=1, inplace=True)

    new_df = pd.get_dummies(df, columns=[attr], prefix=attr)

    return new_df
def calcRanking(df, ranking_df):

    for index, row in df.iterrows():

        season, first, second = getTeam(df, index)

        lower_ranking = ranking_df[(ranking_df['Season'] == int(season)) & (ranking_df['TeamID'] == int(first))]['Avg_rank'].values[0]

        higher_ranking = ranking_df[(ranking_df['Season'] == int(season)) & (ranking_df['TeamID'] == int(second))]['Avg_rank'].values[0]

        df.loc[index, 'lower_ranking'] = lower_ranking

        df.loc[index, 'higher_ranking'] = higher_ranking

    print('Done!')
def rankingSysInfo(rankingSys,teamID):

    day_eval = {}

    team_rank = {}

    groupByTeam = rankings.loc[(rankings['SystemName'] == rankingSys)&(rankings['TeamID'] == teamID)].sort_values('RankingDayNum').groupby(['Season'])

    for name, group in groupByTeam:

        team_rank[name-2014] = group['OrdinalRank']

        day_eval[name-2014] = group['RankingDayNum']

    return day_eval, team_rank
def plotAcrossSeasons(season,teamID):

    plt.figure(figsize=(20,20))

    for rankingSys in (rankings['SystemName'].unique()):

        if len(rankings.loc[(rankings['SystemName'] == rankingSys)&(rankings['TeamID'] == teamID)&(rankings['Season'] == season)]) != 0:

            day_eval, team_rank = rankingSysInfo(rankingSys,teamID)

            single_season_trend = pd.DataFrame({"Day": day_eval[season-2014], "Rank": team_rank[season-2014]})

            plt.plot(single_season_trend["Day"],single_season_trend["Rank"],label=rankingSys)

    plt.title(season)

    plt.xlabel('Days')

    plt.ylabel('Rank')

    plt.legend()
def gameSum(season, teamid):

    gbg = reg_gbg[season - 2010]

    game_played = gbg[(gbg['WTeamID'] == teamid) | (gbg['LTeamID'] == teamid)]

    num_game = len(game_played)

    score = 0

    off_reb = 0

    def_reb = 0

    turnover = 0

    assist = 0

    #General defense ability

    steal = 0

    block = 0

    #Demonstrate perimeter offense

    fgm3 = 0

    fga3 = 0

    #Demonstrate overall offense

    fgm = 0

    fga = 0

    #Demonstrate ability to draw a foul

    fta = 0

    #Demonstrate ability to control foul trouble

    pf = 0

    #Shooting percentage

    pct3 = 0

    pct = 0

    

    for index, row in game_played.iterrows():

        if row['LTeamID'] == teamid:

            score = score + row['LScore']

            off_reb = off_reb + row['LOR']

            def_reb = def_reb + row['LDR']

            turnover = turnover + row['LTO']

            assist = assist + row['LAst']

            steal = steal + row['LStl']

            block = block + row['LBlk']

            fgm3 = fgm3 + row['LFGM3']

            fga3 = fga3 + row['LFGA3']

            fgm = fgm + row['LFGM']

            fga = fga + row['LFGA']

            pf = pf + row['LPF']

        if row['WTeamID'] == teamid:

            score = score + row['WScore']

            off_reb = off_reb + row['WOR']

            def_reb = def_reb + row['WDR']

            turnover = turnover + row['WTO']

            assist = assist + row['WAst']

            steal = steal + row['WStl']

            block = block + row['WBlk']

            fgm3 = fgm3 + row['WFGM3']

            fga3 = fga3 + row['WFGA3']

            fgm = fgm + row['WFGM']

            fga = fga + row['WFGA']

            pf = pf + row['WPF']

    pct3 = fgm3/ fga3 * 100

    

    return score/num_game, off_reb/num_game, def_reb/num_game, turnover/num_game, assist/num_game, steal/num_game, block/num_game, fga3/num_game, pct3
#This patitions the regular season dataset by year from 2010 to 2019

reg_gbg = generateGbgSeparation(range(2010,2020))
#This is season 2010

reg_gbg[0]
# #This calculates the offensive and defensive efficiency of each team in each regular season and apply the result to the 'team' dataset

# for reg, year in zip(reg_gbg,range(2010,2020)):

#     OEff, DEff = applyEff_team(reg, year)

#     teams['Offensive_Eff_' + str(year)] = OEff

#     teams['Defensive_Eff_' + str(year)] = DEff
# for reg, year in zip([reg_2015, reg_2016, reg_2017, reg_2018, reg_2019], range(2015,2020)):

#     applyEff_season(reg)

#     print(str(year) + 'done!')
# for tourney, year in zip([tourney_2015, tourney_2016, tourney_2017, tourney_2018, tourney_2019], range(2015,2020)):

#     applyEff_season(tourney, year)

#     print(str(year) + 'done!')
# for train in [reg_2015, reg_2016, reg_2017, reg_2018, reg_2019, tourney_2015, tourney_2016, tourney_2017, tourney_2018, tourney_2019]:

#     train['OEff_diff'] = train['Team1_OEff'] - train['Team2_OEff']

#     train['DEff_diff'] = train['Team1_DEff'] - train['Team2_DEff']
reg_season_from_10 = reg_season_gbg.loc[reg_season_gbg['Season'] >= 2010]
reg_season_from_10
base_elo = 1600

elo_mess = reg_season_result

team_ids = set(elo_mess.WTeamID).union(set(elo_mess.LTeamID))

elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))
#These value are able to tuned for a potential better performance

K = 20

HOME_ADVANTAGE = 100
#This calculate the margin of victory

elo_mess['margin'] = elo_mess.WScore - elo_mess.LScore
def elo_pred(elo1, elo2):

    return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))



def expected_margin(elo_diff):

    return((7.5 + 0.006 * elo_diff))



def elo_update(w_elo, l_elo, margin):

    elo_diff = w_elo - l_elo

    pred = elo_pred(w_elo, l_elo)

    mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)

    update = K * mult * (1 - pred)

    return(pred, update)
preds = []

w_elo = []

l_elo = []



# Loop over all rows of the games dataframe

for row in elo_mess.itertuples():

    

    # Get key data from current row

    w = row.WTeamID

    l = row.LTeamID

    margin = row.margin

    wloc = row.WLoc

    

    # Does either team get a home-court advantage?

    w_ad, l_ad, = 0., 0.

    if wloc == "H":

        w_ad += HOME_ADVANTAGE

    elif wloc == "A":

        l_ad += HOME_ADVANTAGE

    

    # Get elo updates as a result of the game

    pred, update = elo_update(elo_dict[w] + w_ad,

                              elo_dict[l] + l_ad, 

                              margin)

    elo_dict[w] += update

    elo_dict[l] -= update

    

    # Save prediction and new Elos for each round

    preds.append(pred)

    w_elo.append(elo_dict[w])

    l_elo.append(elo_dict[l])
def final_elo_per_season(df, team_id):

    d = df.copy()

    d = d.loc[(d.WTeamID == team_id) | (d.LTeamID == team_id), :]

    d.sort_values(['Season', 'DayNum'], inplace=True)

    d.drop_duplicates(['Season'], keep='last', inplace=True)

    w_mask = d.WTeamID == team_id

    l_mask = d.LTeamID == team_id

    d['season_elo'] = None

    d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']

    d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']

    out = pd.DataFrame({

        'team_id': team_id,

        'Season': d.Season,

        'season_elo': d.season_elo

    })

    return(out)
elo_mess['w_elo'] = w_elo

elo_mess['l_elo'] = l_elo
df_list = [final_elo_per_season(elo_mess, id) for id in team_ids]

season_elos = pd.concat(df_list)
season_elos[season_elos['Season'] == 2019]
reg_2019 = elo_mess[(elo_mess['Season'] >= 2015) & (elo_mess['Season'] <= 2019)].sort_values('DayNum')
reg_2018 = elo_mess[(elo_mess['Season'] >= 2014) & (elo_mess['Season'] <= 2018)].sort_values('DayNum')
reg_2017 = elo_mess[(elo_mess['Season'] >= 2013) & (elo_mess['Season'] <= 2017)].sort_values('DayNum')
reg_2016 = elo_mess[(elo_mess['Season'] >= 2012) & (elo_mess['Season'] <= 2016)].sort_values('DayNum')
reg_2015 = elo_mess[(elo_mess['Season'] >= 2011) & (elo_mess['Season'] <= 2015)].sort_values('DayNum')
last_day_ranking = rankings[rankings['RankingDayNum'] == 133]

last_day_ranking
ranking_2015 = last_day_ranking[(last_day_ranking['Season'] <= 2015) & (last_day_ranking['Season'] >= 2011)]
ranking_2016 = last_day_ranking[(last_day_ranking['Season'] <= 2016) & (last_day_ranking['Season'] >= 2012)]
ranking_2017 = last_day_ranking[(last_day_ranking['Season'] <= 2017) & (last_day_ranking['Season'] >= 2013)]
ranking_2018 = last_day_ranking[(last_day_ranking['Season'] <= 2018) & (last_day_ranking['Season'] >= 2014)]
ranking_2019 = last_day_ranking[(last_day_ranking['Season'] <= 2019) & (last_day_ranking['Season'] >= 2015)]
def ranking_helper(src,dest):

    for name, group in src:

        curr_season = name[0]

        curr_team = name[1]

        avg = round(np.sum(group['OrdinalRank']) / len(group['OrdinalRank']),1)

        dest = dest.append({'Season':int(curr_season), 'TeamID':int(curr_team), 'Avg_rank':int(avg)}, ignore_index=True)

    return dest
ranking_2019_group = ranking_2019.groupby(['Season','TeamID'])

last_day_ranking_2019 = pd.DataFrame(columns=['Season', 'TeamID', 'Avg_rank'])

last_day_ranking_2019 = ranking_helper(ranking_2019_group, last_day_ranking_2019)

last_day_ranking_2019
ranking_2018_group = ranking_2018.groupby(['Season','TeamID'])

last_day_ranking_2018 = pd.DataFrame(columns=['Season', 'TeamID', 'Avg_rank'])

last_day_ranking_2018 = ranking_helper(ranking_2018_group, last_day_ranking_2018)

last_day_ranking_2018
ranking_2017_group = ranking_2017.groupby(['Season','TeamID'])

last_day_ranking_2017 = pd.DataFrame(columns=['Season', 'TeamID', 'Avg_rank'])

last_day_ranking_2017 = ranking_helper(ranking_2017_group, last_day_ranking_2017)

last_day_ranking_2017
ranking_2016_group = ranking_2016.groupby(['Season','TeamID'])

last_day_ranking_2016 = pd.DataFrame(columns=['Season', 'TeamID', 'Avg_rank'])

last_day_ranking_2016 = ranking_helper(ranking_2016_group, last_day_ranking_2016)

last_day_ranking_2016
ranking_2015_group = ranking_2015.groupby(['Season','TeamID'])

last_day_ranking_2015 = pd.DataFrame(columns=['Season', 'TeamID', 'Avg_rank'])

last_day_ranking_2015 = ranking_helper(ranking_2015_group, last_day_ranking_2015)

last_day_ranking_2015
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import log_loss

from sklearn.model_selection import cross_val_score
tourney_2019 = tournament_result[(tournament_result['Season'] <= 2019) & (tournament_result['Season'] >= 2015)]

elo_filtered = season_elos[(season_elos['Season'] <= 2019) & (season_elos['Season'] >= 2015)]

a = tourney_2019.merge(elo_filtered, left_on=['WTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'w_elo'})

tourney_2019 = a.merge(elo_filtered, left_on=['LTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'l_elo'})

#Usable output

val_2019 = tourney_2019[tourney_2019['Season'] == 2019]

tourney_2019 = tourney_2019[(tourney_2019['Season'] < 2019) & (tourney_2019['Season'] >= 2015)]

test_2019 = tourney_2019[tourney_2019['Season'] == 2018]
tourney_2018 = tournament_result[(tournament_result['Season'] <= 2018) & (tournament_result['Season'] >= 2014)]

elo_filtered = season_elos[(season_elos['Season'] <= 2018) & (season_elos['Season'] >= 2014)]

a = tourney_2018.merge(elo_filtered, left_on=['WTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'w_elo'})

tourney_2018 = a.merge(elo_filtered, left_on=['LTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'l_elo'})

#Usable output

val_2018 = tourney_2018[tourney_2018['Season'] == 2018]

tourney_2018 = tourney_2018[(tourney_2018['Season'] < 2018) & (tourney_2018['Season'] >= 2014)]

test_2018 = tourney_2018[tourney_2018['Season'] == 2017]
tourney_2017 = tournament_result[(tournament_result['Season'] <= 2017) & (tournament_result['Season'] >= 2013)]

elo_filtered = season_elos[(season_elos['Season'] <= 2017) & (season_elos['Season'] >= 2013)]

a = tourney_2017.merge(elo_filtered, left_on=['WTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'w_elo'})

tourney_2017 = a.merge(elo_filtered, left_on=['LTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'l_elo'})

#Usable output

val_2017 = tourney_2017[tourney_2017['Season'] == 2017]

tourney_2017 = tourney_2017[(tourney_2017['Season'] < 2017) & (tourney_2017['Season'] >= 2013)]

test_2017 = tourney_2017[tourney_2017['Season'] == 2016]
tourney_2016 = tournament_result[(tournament_result['Season'] <= 2016) & (tournament_result['Season'] >= 2012)]

elo_filtered = season_elos[(season_elos['Season'] <= 2016) & (season_elos['Season'] >= 2012)]

a = tourney_2016.merge(elo_filtered, left_on=['WTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'w_elo'})

tourney_2016 = a.merge(elo_filtered, left_on=['LTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'l_elo'})

#Usable output

val_2016 = tourney_2016[tourney_2016['Season'] == 2016]

tourney_2016 = tourney_2016[(tourney_2016['Season'] < 2016) & (tourney_2016['Season'] >= 2012)]

test_2016 = tourney_2016[tourney_2016['Season'] == 2015]
tourney_2015 = tournament_result[(tournament_result['Season'] <= 2015) & (tournament_result['Season'] >= 2011)]

elo_filtered = season_elos[(season_elos['Season'] <= 2015) & (season_elos['Season'] >= 2011)]

a = tourney_2015.merge(elo_filtered, left_on=['WTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'w_elo'})

tourney_2015 = a.merge(elo_filtered, left_on=['LTeamID','Season'], right_on=['team_id', 'Season']).drop(['team_id'], axis=1).rename(columns={'season_elo':'l_elo'})

#Usable output

val_2015 = tourney_2015[tourney_2015['Season'] == 2015]

tourney_2015 = tourney_2015[(tourney_2015['Season'] < 2015) & (tourney_2015['Season'] >= 2011)]

test_2015 = tourney_2015[tourney_2015['Season'] == 2014]
voting_clf = VotingClassifier(estimators=[

    ('log_clf', LogisticRegression(penalty='l2', fit_intercept=False, C=0.0001,

                         verbose=False, max_iter=1000, solver='lbfgs')),

    ('svm_clf',SVC(probability=True)),

    ('dt_clf',DecisionTreeClassifier(random_state=666))], voting='soft')
for reg, tourney, val in zip([reg_2015, reg_2016, reg_2017, reg_2018, reg_2019],[tourney_2015, tourney_2016, tourney_2017, tourney_2018, tourney_2019],[val_2015, val_2016, val_2017, val_2018, val_2019]):

    featureAddition(reg, elo=True)

    featureAddition(tourney, elo=True)

    featureAddition(val, elo=True)
calcRanking(reg_2015, last_day_ranking_2015)

calcRanking(tourney_2015, last_day_ranking_2015)



calcRanking(reg_2016, last_day_ranking_2016)

calcRanking(tourney_2016, last_day_ranking_2016)



calcRanking(reg_2017, last_day_ranking_2017)

calcRanking(tourney_2017, last_day_ranking_2017)



calcRanking(reg_2018, last_day_ranking_2018)

calcRanking(tourney_2018, last_day_ranking_2018)



calcRanking(reg_2019, last_day_ranking_2019)

calcRanking(tourney_2019, last_day_ranking_2019)
feature = ['log_lower_elo','log_higher_elo']

# feature = ['log_lower_elo','log_higher_elo','log_lower_ranking','log_higher_ranking']
# reg_2015['higher_elo_band'] = pd.cut(reg_2015['higher_elo'],3)

# reg_2015['higher_elo_band'].unique()
# interval_1 = pd.Interval(left=832.325,right=1262.732)

# interval_2 = pd.Interval(left=1262.732,right=1691.853)

# interval_3 = pd.Interval(left=1691.853,right=2120.973)
# reg_2015.loc[(reg_2015['higher_elo_band'] == interval_1),'higher_elo'] = 1

# reg_2015.loc[(reg_2015['higher_elo_band'] == interval_2),'higher_elo'] = 2

# reg_2015.loc[(reg_2015['higher_elo_band'] == interval_3),'higher_elo'] = 3



# reg_2015['higher_elo'] = reg_2015['higher_elo'].astype(int)

# reg_2015.drop('higher_elo_band', axis=1, inplace=True)



# reg_2015 = pd.get_dummies(reg_2015, columns=['higher_elo'], prefix='higher_elo')
#Standardization on elo

for df in [reg_2015, reg_2016, reg_2017, reg_2018, reg_2019, val_2015, val_2016, val_2017, val_2018, val_2019]:

    df['standardized_lower_elo'] = (df['lower_elo'] - df['lower_elo'].mean()) / df['lower_elo'].std()

    df['standardized_higher_elo'] = (df['higher_elo'] - df['higher_elo'].mean()) / df['higher_elo'].std()
# #Standardization on ranking

# for df in [reg_2015, reg_2016, reg_2017, reg_2018, reg_2019, val_2015, val_2016, val_2017, val_2018, val_2019]:

#     df['standardized_lower_ranking'] = (df['lower_ranking'] - df['lower_ranking'].mean()) / df['lower_ranking'].std()

#     df['standardized_higher_ranking'] = (df['higher_ranking'] - df['higher_ranking'].mean()) / df['higher_ranking'].std()
#Log transform on elo

for df in [reg_2015, reg_2016, reg_2017, reg_2018, reg_2019, val_2015, val_2016, val_2017, val_2018, val_2019]:

    df['log_lower_elo'] = (df['standardized_lower_elo']-df['standardized_lower_elo'].min()+1) .transform(np.log)

    df['log_higher_elo'] = (df['standardized_higher_elo']-df['standardized_higher_elo'].min()+1) .transform(np.log)
# #Log transform on ranking

# for df in [reg_2015, reg_2016, reg_2017, reg_2018, reg_2019, val_2015, val_2016, val_2017, val_2018, val_2019]:

#     df['log_lower_ranking'] = (df['standardized_lower_elo']-df['standardized_lower_elo'].min()+1) .transform(np.log)

#     df['log_higher_ranking'] = (df['standardized_higher_elo']-df['standardized_higher_elo'].min()+1) .transform(np.log)
plt.figure(figsize=(9,3))

reg_2015['standardized_higher_elo'].hist()

plt.title('standardized_higher_elo')

plt.show()

# plt.figure(figsize=(9,3))

# reg_2015['lower_ranking'].hist()

# plt.figure(figsize=(9,3))

# reg_2015['log_lower_ranking'].hist()

plt.figure(figsize=(9,3))

plt.title('standardized_higher_elo')

reg_2015['higher_elo'].hist()

plt.show()
tourney_seed = seedSeparation([2015,2016,2017,2018,2019])
from itertools import combinations

from itertools import permutations
def generateTourneySubFile(year):

    id_list = []

    comb = combinations(tourney_seed[year-2015]['TeamID'],2)

    for i in list(comb):

        firstTeam = i[0]

        secondTeam = i[1]

        if firstTeam > secondTeam:

            firstTeam = i[1]

            secondTeam = i[0]

        id_list.append(str(year) + '_' + str(firstTeam) + '_' + str(secondTeam))

    df = pd.DataFrame({'ID':id_list})

    return df
pred_2015 = generateTourneySubFile(2015)

pred_2016 = generateTourneySubFile(2016)

pred_2017 = generateTourneySubFile(2017)

pred_2018 = generateTourneySubFile(2018)

pred_2019 = generateTourneySubFile(2019)
elo_2015 = season_elos.loc[season_elos['Season'] == 2015]

elo_2016 = season_elos.loc[season_elos['Season'] == 2016]

elo_2017 = season_elos.loc[season_elos['Season'] == 2017]

elo_2018 = season_elos.loc[season_elos['Season'] == 2018]

elo_2019 = season_elos.loc[season_elos['Season'] == 2019]
def appendElo(df_pred, df_elo):

    for index, row in df_pred.iterrows():

        season, first, second = getTeam(df_pred, index)

        first_team_elo = round(float(df_elo.loc[df_elo['team_id'] == int(first)]['season_elo'].values),2)

        second_team_elo = round(float(df_elo.loc[df_elo['team_id'] == int(second)]['season_elo'].values),2)

        df_pred.loc[index, 'lower_elo'] = first_team_elo

        df_pred.loc[index, 'higher_elo'] = second_team_elo
for pred,elo, val,ranking in zip([pred_2015, pred_2016, pred_2017, pred_2018, pred_2019],[elo_2015, elo_2016, elo_2017, elo_2018, elo_2019],[val_2015,val_2016,val_2017,val_2018,val_2019],[last_day_ranking_2015, last_day_ranking_2016,last_day_ranking_2017,last_day_ranking_2018,last_day_ranking_2019]):

    appendElo(pred, elo)

    calcRanking(pred, ranking)

    calcRanking(val, ranking)
#Standardization

for df in [pred_2015, pred_2016, pred_2017, pred_2018, pred_2019]:

#     df['standardized_lower_ranking'] = (df['lower_ranking'] - df['lower_ranking'].mean()) / df['lower_ranking'].std()

#     df['standardized_higher_ranking'] = (df['higher_ranking'] - df['higher_ranking'].mean()) / df['higher_ranking'].std()

    df['standardized_lower_elo'] = (df['lower_elo'] - df['lower_elo'].mean()) / df['lower_elo'].std()

    df['standardized_higher_elo'] = (df['higher_elo'] - df['higher_elo'].mean()) / df['higher_elo'].std()
#Log transform

for df in [pred_2015, pred_2016, pred_2017, pred_2018, pred_2019]:

    df['log_lower_elo'] = (df['standardized_lower_elo']-df['standardized_lower_elo'].min()+1) .transform(np.log)

    df['log_higher_elo'] = (df['standardized_higher_elo']-df['standardized_higher_elo'].min()+1) .transform(np.log)

#     df['log_lower_ranking'] = (df['standardized_lower_ranking']-df['standardized_lower_ranking'].min()+1) .transform(np.log)

#     df['log_higher_ranking'] = (df['standardized_higher_ranking']-df['standardized_higher_ranking'].min()+1) .transform(np.log)
# for df, year in zip([pred_2015, pred_2016, pred_2017, pred_2018, pred_2019],range(2015,2020)):

#     applyEff_season(df,year)

#     df['OEff_diff'] = df['Team1_OEff'] - df['Team2_OEff']

#     df['DEff_diff'] = df['Team1_DEff'] - df['Team2_DEff']

#     df.drop(['Team1_OEff','Team1_DEff','Team2_OEff','Team2_DEff'], axis=1, inplace=True)
# temp_2015 = pd.concat([reg_2015, test_2015],ignore_index=True)

# print('Feature addition....')

# featureAddition(val_2015)

# print('Done!')

X_train = reg_2015[feature]

y_train = reg_2015[['lower_win']]

print('Fitting....')

voting_clf.fit(X_train, y_train)

print('Done!')

appendPred(pred_2015)

# pred_2015.drop(['lower_elo','higher_elo','lower_ranking','higher_ranking'],axis=1, inplace=True)

print(log_loss(val_2015['lower_win'], voting_clf.predict_proba(val_2015[feature])[:,1]))
# featureAddition(test_2016, elo=True)

# temp_2016 = pd.concat([reg_2016, test_2016],ignore_index=True)

X_train = reg_2016[feature]

y_train = reg_2016[['lower_win']]

voting_clf.fit(X_train, y_train)

appendPred(pred_2016)

# pred_2016.drop(feature,axis=1, inplace=True)

print(log_loss(val_2016['lower_win'], voting_clf.predict_proba(val_2016[feature])[:,1]))
# featureAddition(test_2017, elo=True)

# temp_2017 = pd.concat([reg_2017, test_2017],ignore_index=True)

X_train = reg_2017[feature]

y_train = reg_2017[['lower_win']]

voting_clf.fit(X_train, y_train)

appendPred(pred_2017)

# pred_2017.drop(feature,axis=1, inplace=True)

print(log_loss(val_2017['lower_win'], voting_clf.predict_proba(val_2017[feature])[:,1]))
# featureAddition(test_2018, elo=True)

# temp_2018 = pd.concat([reg_2018, test_2018],ignore_index=True)

X_train = reg_2018[feature]

y_train = reg_2018[['lower_win']]

voting_clf.fit(X_train, y_train)

appendPred(pred_2018)

# pred_2018.drop(feature,axis=1, inplace=True)

print(log_loss(val_2018['lower_win'], voting_clf.predict_proba(val_2018[feature])[:,1]))
# featureAddition(test_2019, elo=True)

# temp_2019 = pd.concat([reg_2019, test_2019],ignore_index=True)

X_train = reg_2019[feature]

y_train = reg_2019[['lower_win']]

voting_clf.fit(X_train, y_train)

appendPred(pred_2019)

# pred_2019.drop(feature,axis=1, inplace=True)

print(log_loss(val_2019['lower_win'], voting_clf.predict_proba(val_2019[feature])[:,1]))
submission_file = pd.concat([pred_2015, pred_2016, pred_2017, pred_2018, pred_2019])

for index, row in submission_file.iterrows():

    if row['Pred'] > 0.95:

        submission_file.loc[index,'Pred'] = 0.95

    if row['Pred'] < 0.05:

        submission_file.loc[index,'Pred'] = 0.05
submission_file = submission_file[['ID','Pred']]
submission_file.to_csv('submission.csv',index=False)
gameReport('2019_1192_1293')
#Get team Name

def gameReport(ID):

    field = {'Team Name','Last Day Ranking','Avg Score','AScore allowed','Avg OR','Avg DR'}

    comparison = pd.DataFrame(columns={'Lower team', 'Higher team'}, index=field)

    season = int(ID[0:4])

    first = int(ID[5:9])

    second = int(ID[10:14])

    lower_team_name = teams[teams['TeamID'] == first]['TeamName'].values[0]

    higher_team_name = teams[teams['TeamID'] == second]['TeamName'].values[0]

    comparison.loc['Team Name'] = [lower_team_name, higher_team_name]

    

    #Last day ranking

    lower_rank = last_day_ranking_2019[(last_day_ranking_2019['TeamID'] == first) & (last_day_ranking_2019['Season'] == season)]['Avg_rank'].values[0]

    higher_rank = last_day_ranking_2019[(last_day_ranking_2019['TeamID'] == second) & (last_day_ranking_2019['Season'] == season)]['Avg_rank'].values[0]

    comparison.loc['Last Day Ranking'] = [lower_rank, higher_rank]

    

    #Avg game stat

    l_score, l_off_reb, l_def_reb, l_to, l_asst, l_stl, l_blk, l_fga3, l_pct3= gameSum(season, first)

    h_score, h_off_reb, h_def_reb, h_to, h_asst, h_stl, h_blk, h_fga3, h_pct3= gameSum(season, second)

    comparison.loc['Avg Score'] = [l_score,h_score]

    comparison.loc['Avg OR'] = [l_off_reb, h_off_reb]

    comparison.loc['Avg DR'] = [l_def_reb, h_off_reb]

    comparison.loc['Avg TO'] = [l_to, h_to]

    comparison.loc['Avg Asst'] = [l_asst, h_asst]

    comparison.loc['Avg Stl'] = [l_stl, h_stl]

    comparison.loc['Avg Blk'] = [l_blk, h_blk]

    comparison.loc['Avg FGA3'] = [l_fga3, h_fga3]

    comparison.loc['3PT PCT'] = [l_pct3, h_pct3]

    

    

    #Ranking trend

#     plotAcrossSeasons(season, first)

#     plotAcrossSeasons(season, second)

    

    return comparison