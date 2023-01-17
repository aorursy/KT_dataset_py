from collections import Counter

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')



curr_dir = '../input/mens-machine-learning-competition-2018/' 

tourney_games = pd.read_csv(curr_dir + 'NCAATourneyCompactResults.csv')

regular_games = pd.read_csv(curr_dir + 'RegularSeasonCompactResults.csv')

test_games = pd.read_csv(curr_dir + 'SampleSubmissionStage2.csv')

seeds = pd.read_csv(curr_dir + 'NCAATourneySeeds.csv')

massey_ordinals = pd.read_csv(curr_dir + 'MasseyOrdinals_thruSeason2018_Day128.csv')
test_games.head()
test_games.shape 
68*67/2 # 68 choose 2
def test_features(games):

    (games['Season'], games['Team1'], games['Team2']) = zip(*games.ID.apply(lambda x: tuple(map(int, x.split('_')))))

    games['DayNum'] = 134

    cols_to_keep = ['Season', 'Team1', 'Team2', 'DayNum']

    games = games[cols_to_keep]

    return games

test_games = test_features(test_games)
test_games.head()
tourney_games.head()
regular_games.head()
def basic_features(games):

    (games['Team1'], games['Team2']) = np.where(games.WTeamID < games.LTeamID, (games.WTeamID, games.LTeamID), (games.LTeamID, games.WTeamID))

    games['Prediction'] = np.where(games.WTeamID==games.Team1, 1, 0)

    games['Score_difference'] = np.where(games.WTeamID==games.Team1, games.WScore - games.LScore, games.LScore - games.WScore)

    cols = ['Season', 'Team1', 'Team2', 'DayNum', 'Score_difference', 'Prediction']

    games = games[cols]

    return games

tourney_games = basic_features(tourney_games)

regular_games = basic_features(regular_games)
tourney_games.head()
regular_games.head()
tourney_games.shape, regular_games.shape
print("Total number of games to be used in training:", tourney_games.shape[0] + regular_games.shape[0])
seeds.head()
seeds = seeds.set_index(['Season', 'TeamID'])

seeds = seeds['Seed'].to_dict()

type(seeds)
def seed_features(games):

    games['Seed_diff'] = games.apply(lambda row: int(seeds[(row.Season, row.Team1)][1:3]) -

                                                int(seeds[(row.Season, row.Team2)][1:3]), axis=1)

    return games

test_games = seed_features(test_games)

tourney_games = seed_features(tourney_games)
tourney_games.head()
massey_ordinals.head()
massey_ordinals[(massey_ordinals.TeamID ==1101) & (massey_ordinals.Season == 2014) & (massey_ordinals.RankingDayNum ==9)] 
massey_ordinals = massey_ordinals.groupby(['TeamID', 'Season', 'RankingDayNum']).median()

massey_ordinals.head()
ordinals_dict = massey_ordinals['OrdinalRank'].to_dict()



def massey_ranking_difference(Team1, Team2, Season, DayNum):

    if Season < 2003:

        return np.nan

    try:

        Ranking1 = ordinals_dict[(Team1, Season, DayNum)]

    except:

        try:

            RankingDays1 = massey_ordinals.loc[Team1, Season].index

            LatestDayTeam1 = RankingDays1[RankingDays1 <= DayNum][-1]

            Ranking1 = ordinals_dict[(Team1, Season, LatestDayTeam1)]

        except: return np.nan

    try:

        Ranking2 = ordinals_dict[(Team2, Season, DayNum)]

    except:

        try:

            RankingDays2 = massey_ordinals.loc[Team2, Season].index

            LatestDayTeam2 = RankingDays2[RankingDays2 <= DayNum][-1]

            Ranking2 = ordinals_dict[(Team2, Season, LatestDayTeam2)]

        except: return np.nan

    return Ranking1 - Ranking2



def ranking_feature(games, test=False):

    if test:

        games['Ranking_diff'] = games.apply(lambda row: 

                    massey_ranking_difference(row.Team1, row.Team2, 2018, 128), axis=1)

        

    else:

        games['Ranking_diff'] = games.apply(lambda row: 

                    massey_ranking_difference(row.Team1, row.Team2, row.Season, row.DayNum), axis=1)

    return games



tourney_games = ranking_feature(tourney_games)

regular_games = ranking_feature(regular_games)

test_games = ranking_feature(test_games, test=True)
tourney_games.tail()
games = tourney_games.set_index('Season').groupby('Season')
games.describe()
tourney_games_count = {1985: {}} # dictionary of season-wise dictionaries



for grp in games:

    season = grp[0]+1

    df = pd.concat([grp[1].Team1.value_counts(), grp[1].Team2.value_counts()], axis=1).fillna(0)

    df['Season'] = season

    df['games_played'] = df.Team1 + df.Team2

    current_count = df['games_played'].to_dict()

    total_count = Counter(tourney_games_count[season-1]) + Counter(current_count)

    tourney_games_count[season] = total_count



def games_played_difference(Team1, Team2, season):

    games_played_Team1 = tourney_games_count[season].get(Team1, 0)

    games_played_Team2 = tourney_games_count[season].get(Team2, 0)

    return round((games_played_Team1 - games_played_Team2)/(season-1984), 2)



def games_played_feature(games):

    games['Tourney_games_played_diff'] = games.apply(lambda row: 

                            games_played_difference(row.Team1, row.Team2, row.Season), axis=1)

    games['Tournament'] = 1

    return games



tourney_games = games_played_feature(tourney_games)

test_games = games_played_feature(test_games)

regular_games['Tournament'] = 0
all_games = pd.concat([tourney_games, regular_games]).copy()

all_games.sort_values(['Season', 'DayNum'], inplace=True)

hash_scores = {}

b = 0.8

def scores(row):

    if (row.Team1, row.Team2) in hash_scores:

        previous_average_score_difference = hash_scores[(row.Team1, row.Team2)]

        average_score_difference = b*previous_average_score_difference + (1-b)*row.Score_difference

    else: 

        previous_average_score_difference = np.nan 

        average_score_difference = row.Score_difference

    hash_scores[(row.Team1, row.Team2)] = average_score_difference

    return previous_average_score_difference
all_games['Average_score_difference'] = all_games.apply(lambda row: scores(row), axis=1)

all_games.set_index(['Team1', 'Team2', 'Season', 'DayNum'], inplace=True)

all_games.sample(10)
all_games = all_games['Average_score_difference'].to_dict()

def score_difference_feature(games, test=False):  

    if test:

        games['Average_score_diff'] = games.apply(lambda row: 

                            hash_scores.get((row.Team1, row.Team2), np.nan), axis=1)

    else:

        games['Average_score_diff'] = games.apply(lambda row: 

                            all_games[(row.Team1, row.Team2, row.Season, row.DayNum)], axis=1)

    return games



tourney_games = score_difference_feature(tourney_games)

regular_games = score_difference_feature(regular_games)

test_games = score_difference_feature(test_games, test=True)
tourney_games.columns
def final_features(games):

    games.fillna(0, inplace=True)

    games['Team1'] = games['Team1'].astype('category', ordered=False)

    games['Team2'] = games['Team2'].astype('category', ordered=False)

    features_to_keep = ['Team1', 'Team2', 'Seed_diff', 'Average_score_diff', 'Tourney_games_played_diff', 

                        'Ranking_diff', 'Tournament'] 

    games = games[features_to_keep]

    return games
train = pd.concat([tourney_games, regular_games])

prediction = train.Prediction

test = test_games



train = final_features(train)

test = final_features(test)
train.iloc[2000:2010]
test.head()