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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
main_data_dir = '../input/mens-machine-learning-competition-2018/'
elo_data_dir = '../input/fivethirtyeight-elo-ratings/'

# Any results you write to the current directory are saved as output.
print(check_output(["ls", main_data_dir]).decode("utf8"))
df_seeds = pd.read_csv(main_data_dir + 'NCAATourneySeeds.csv')
df_tour = pd.read_csv(main_data_dir + 'NCAATourneyCompactResults.csv')
df_elo_ratings = pd.read_csv(elo_data_dir + 'season_elos.csv')
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
df_concat['ELODiff'] = df_concat.WTeamELO - df_concat.LTeamELO
df_concat.head()
df_concat.tail()
# Sanity check
df_concat[(df_concat['Season'] == 2003) & (df_concat['WTeamID'] == 1112) & (df_concat['LTeamID'] == 1436)]
df_wins = pd.DataFrame()
df_wins['ELODiff'] = df_concat['ELODiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['ELODiff'] = -df_concat['ELODiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))
df_predictions.head()
print(df_predictions.shape)
X_train = df_predictions.ELODiff.values.reshape(-1,1)
print(X_train.shape)
y_train = df_predictions.Result.values
print(y_train)
X_train, y_train = shuffle(X_train, y_train)
y_train.shape
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=10)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
df_sample_sub = pd.read_csv(main_data_dir + 'SampleSubmissionStage2.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))
# Confirm that the df_elo_ratings still has complete (1985-2018) data
df_elo_ratings.Season.sort_values().unique()
# Rename team ID column back to the generic form
df_elo_ratings = df_elo_ratings.rename(columns={'LTeamID': 'TeamID'})
df_elo_ratings.tail()
X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_elo = df_elo_ratings[(df_elo_ratings.TeamID == t1) & (df_elo_ratings.Season == year)].season_elo.values[0]
    t2_elo = df_elo_ratings[(df_elo_ratings.TeamID == t2) & (df_elo_ratings.Season == year)].season_elo.values[0]
    diff_elo = t1_elo - t2_elo
    X_test[ii, 0] = diff_elo
preds = clf.predict_proba(X_test)[:,1]
df_sample_sub.Pred = preds
df_sample_sub.head()
df_sample_sub.to_csv('dan_douthit_elo_predictions_2018.csv', index=False)
