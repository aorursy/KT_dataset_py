# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('mode.chained_assignment', None)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
main_data_dir = '../input/mens-machine-learning-competition-2018/'
elo_data_dir = '../input/fivethirtyeight-elo-ratings/'
avg_data_dir = '../input/rolling-averages-for-pre-tournament-games/'

# Any results you write to the current directory are saved as output.
df_average_data = pd.read_csv(avg_data_dir + 'rolling_average_data.csv')
df_seeds = pd.read_csv(main_data_dir + 'NCAATourneySeeds.csv')
df_tour = pd.read_csv(main_data_dir + 'NCAATourneyCompactResults.csv')
df_elo_ratings = pd.read_csv(elo_data_dir + 'season_elos.csv')
df_average_data.tail()
df_elo_ratings = df_elo_ratings.rename(columns={'team_id':'WTeamID', 'season': 'Season'})
df_elo_ratings.head()
df_elo_ratings[(df_elo_ratings['Season'] == 1985) & (df_elo_ratings['WTeamID'] == 1325)]
def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tour.head()
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat.head()
df_concat = pd.merge(left=df_concat, right=df_elo_ratings, how='left', on=['Season', 'WTeamID'])
df_concat = df_concat.rename(columns={'season_elo': 'WTeamELO'})
df_elo_ratings = df_elo_ratings.rename(columns={'WTeamID': 'LTeamID'})
df_concat = pd.merge(left=df_concat, right=df_elo_ratings, how='left', on=['Season', 'LTeamID'])
df_concat = df_concat.rename(columns={'season_elo': 'LTeamELO'})
df_concat.tail()
# Sanity check to make sure data is consistent between all tourney games and "KPom era" games
df_concat[(df_concat['Season'] == 2003) & (df_concat['WTeamID'] == 1112) & (df_concat['LTeamID'] == 1436)]
# In this naive version, drop the rows corresponding to years being predicted (2014-2017). These can be added back in later.
df_concat = df_concat[df_concat['Season'] < 2014]
# Drop the seed related columns as we aren't using them in this model
df_concat.drop(labels=['WSeed', 'LSeed', 'SeedDiff'], inplace=True, axis=1)
# Also drop the rows for which we don't have rolling averages
df_concat = df_concat[df_concat['Season'] > 2002]
df_concat.tail()
df_average_data.tail()
# identify columns with NaN (dirty free throw data) and "normalize" to 70%
df_average_data[df_average_data.isnull().any(axis=1)]
values= {'FTPAvg': 70.0}
df_average_data = df_average_data.fillna(value=values)
df_average_data[df_average_data.isnull().any(axis=1)]
#X_train = np.zeros(shape=(n_test_games, 10))
X_train = []
y_train = []
# find end of season ELO ratings and regular season rolling averages for each tournament game played
for ii, row in df_concat.iterrows():
    win_team_features = []
    lose_team_features = []
    win_elo = row.WTeamELO
    lose_elo = row.LTeamELO
    win_team_features.append(win_elo)
    lose_team_features.append(lose_elo)
    # don't want to append the season and team ID here
    win_team_avgs = df_average_data[(df_average_data.Season == row.Season) & (df_average_data.TeamID == row.WTeamID)].iloc[0].values[2:]
    for average in win_team_avgs:
        win_team_features.append(average)
        
    lose_team_avgs = df_average_data[(df_average_data.Season == row.Season) & (df_average_data.TeamID == row.WTeamID)].iloc[0].values[2:]
    for average in lose_team_avgs:
        lose_team_features.append(average)
    
    # Randomly select win and lose order to train for both classes (0 and 1)
    if random.random() > 0.5:
        X_train.append(win_team_features + lose_team_features)
        y_train.append(1)
    else:
        X_train.append(lose_team_features + win_team_features)
        y_train.append(0)
        
# Sanity check
print("X_train length is: " + str(len(X_train)))
print("y_train length is: " + str(len(y_train)))
print("First item in X_train vector is:")
X_train[0]
X_train, y_train = shuffle(X_train, y_train)
len(X_train)
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=10)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
df_sample_sub = pd.read_csv(main_data_dir + 'SampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))
# Confirm that the df_elo_ratings still has complete (1985-2017) data
df_elo_ratings.Season.sort_values().unique()
# Rename team ID column back to the generic form
df_elo_ratings = df_elo_ratings.rename(columns={'LTeamID': 'TeamID'})
df_elo_ratings.tail()
X_test = []
for ii, row in df_sample_sub.iterrows():
    team1_features = []
    team2_features = []
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_elo = df_elo_ratings[(df_elo_ratings.TeamID == t1) & (df_elo_ratings.Season == year)].season_elo.values[0]
    t1_avgs = df_average_data[(df_average_data.Season == year) & (df_average_data.TeamID == t1)].iloc[0].values[2:]
    t2_elo = df_elo_ratings[(df_elo_ratings.TeamID == t2) & (df_elo_ratings.Season == year)].season_elo.values[0]
    t2_avgs = df_average_data[(df_average_data.Season == year) & (df_average_data.TeamID == t2)].iloc[0].values[2:]
    team1_features.append(t1_elo)
    for average in t1_avgs:
        team1_features.append(average)
    team2_features.append(t2_elo)
    for average in t2_avgs:
        team2_features.append(average)
    X_test.append(team1_features + team2_features)
len(X_test)
preds = clf.predict_proba(X_test)[:,1]
df_sample_sub.Pred = preds
df_sample_sub.head()
# Create function to look up ground truth values for predictions
def lookup_truth(idString):
    year, t1, t2 = get_year_t1_t2(idString)
    tour_row = df_tour[(df_tour.Season == year) & (df_tour.WTeamID.isin([t1,t2])) & (df_tour.LTeamID.isin([t1, t2]))]
    # game didn't happen
    if len(tour_row) == 0:
        return -1
    elif tour_row.WTeamID.values[0] == t1:  
        return 1  # truth is won game
    else:
        return 0  # truth is lost game
# Add truth labels to predictions and drop rows for hypothetical matchups that weren't actually played
df_sample_sub_with_truth = df_sample_sub.copy()
df_sample_sub_with_truth['truth'] = df_sample_sub_with_truth.ID.apply(lookup_truth)
df_sample_sub_with_truth = df_sample_sub_with_truth[df_sample_sub_with_truth.truth != -1]

# Calculate and display the log loss
logLoss = log_loss(y_true=df_sample_sub_with_truth['truth'], y_pred=df_sample_sub_with_truth['Pred'])
print(str(logLoss))
#df_sample_sub.to_csv('dan_douthit_elo_recent_stats_predictions.csv', index=False)
