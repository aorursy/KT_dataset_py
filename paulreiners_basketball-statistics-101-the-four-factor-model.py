# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_dir = '../input/datafiles/'

winning_teams_df = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
def compute_efg(df, winner=True):

    df['WEFG'] = (df['WFGM'] + 0.5 * df['WFGM3']) / df['WFGA']

    df['LEFG'] = (df['LFGM'] + 0.5 * df['LFGM3']) / df['LFGA']

    df.drop(labels=['WFGM', 'WFGM3', 'WFGA3', 'LFGM', 'LFGM3', 'LFGA3'], inplace=True, axis=1)

    if winner:

        df.columns = ['Season', 'Team', 'WFGA', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WTO', 'LFGA', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LTO', 'OffensiveShooting', 'DefensiveShooting']

    else:

        df.columns = ['Season', 'Team', 'WFGA', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WTO', 'LFGA', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LTO', 'DefensiveShooting', 'OffensiveShooting']

    column_titles = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'WFGA', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WTO',

           'LFGA', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LTO']

    df = df.reindex(columns=column_titles)



    return df
def compute_tpp(df, winner=True):

    # Turnover Rate = Turnovers / (Field Goal Attempts + 0.44*Free Throw Attempts + Turnovers)

    df['WTPP'] = df['WTO'] / (df['WFGA'] + 0.44 * df['WFTA'] + df['WTO'])

    df['LTPP'] = df['LTO'] / (df['LFGA'] + 0.44 * df['LFTA'] + df['LTO'])

    df.drop(labels=['WTO', 'LTO'], inplace=True, axis=1)

    if winner:

        df.columns = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'WFGA', 'WFTM', 'WFTA', 'WOR', 'WDR', 'LFGA', 'LFTM', 

                      'LFTA', 'LOR', 'LDR', 'OffensiveTOs', 'DefensiveTOs']

    else:

        df.columns = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'WFGA', 'WFTM', 'WFTA', 'WOR', 'WDR', 'LFGA', 'LFTM', 

                      'LFTA', 'LOR', 'LDR', 'DefensiveTOs', 'OffensiveTOs']

    column_titles = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'OffensiveTOs', 'DefensiveTOs', 'WFGA', 'WFTM', 'WFTA', 

                     'WOR', 'WDR', 'LFGA', 'LFTM', 'LFTA', 'LOR', 'LDR']

    df = df.reindex(columns=column_titles)



    return df
def compute_orp(df, winner=True):

    df['ORP'] = df['WOR'] / (df['WOR'] + df['LDR'])

    df['DRP'] = df['WDR'] / (df['WDR'] + df['LOR'])

    df.drop(labels=['WOR', 'WDR', 'LDR', 'LOR'], inplace=True, axis=1)

    if winner:

        df.columns = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'OffensiveTOs', 'DefensiveTOs', 'WFGA', 'WFTM', 'WFTA', 

                      'LFGA', 'LFTM', 'LFTA', 'OffensiveRebounding', 'DefensiveRebounding']

    else:

        df.columns = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'OffensiveTOs', 'DefensiveTOs', 'WFGA', 'WFTM', 'WFTA', 

                      'LFGA', 'LFTM', 'LFTA', 'DefensiveRebounding', 'OffensiveRebounding']

    column_titles = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'OffensiveTOs', 'DefensiveTOs', 'OffensiveRebounding', 

                     'DefensiveRebounding', 'WFGA', 'WFTM', 'WFTA', 'LFGA', 'LFTM', 'LFTA']

    df = df.reindex(columns=column_titles)



    return df
def compute_ftr(df, winner=True):

    df['WFTR'] = df['WFTM'] / df['WFGA']

    df.drop(labels=['WFTM', 'WFGA', 'WFTA'], inplace=True, axis=1)

    df['LFTR'] = df['LFTM'] / df['LFGA']

    df.drop(labels=['LFTM', 'LFGA', 'LFTA'], inplace=True, axis=1)

    if winner:

        df.columns = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'OffensiveTOs', 'DefensiveTOs', 'OffensiveRebounding', 

                      'DefensiveRebounding', 'OffensiveFTs', 'DefensiveFTs']

    else:

        df.columns = ['Season', 'Team', 'OffensiveShooting', 'DefensiveShooting', 'OffensiveTOs', 'DefensiveTOs', 'OffensiveRebounding', 

                      'DefensiveRebounding', 'DefensiveFTs', 'OffensiveFTs']



    return df
def compute_four_factors(df, winner=True):

    if winner:

        df.drop(labels=['DayNum', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WAst', 'LAst', 'WStl', 'LStl', 'WBlk', 'LBlk', 'WPF', 'LPF'], inplace=True, axis=1)

    else:

        df.drop(labels=['DayNum', 'WScore', 'WTeamID', 'LScore', 'WLoc', 'NumOT', 'WAst', 'LAst', 'WStl', 'LStl', 'WBlk', 'LBlk', 'WPF', 'LPF'], inplace=True, axis=1)

    winning_with_efg = compute_efg(df, winner)

    winning_with_tpp = compute_tpp(winning_with_efg, winner)

    winning_with_orp = compute_orp(winning_with_tpp, winner)

    winning_with_ftr = compute_ftr(winning_with_orp, winner)

    if winner:

        winning_with_ftr['won'] = 1

    else:

        winning_with_ftr['won'] = 0



    return winning_with_ftr
four_factors_df = compute_four_factors(winning_teams_df)
losing_teams_df = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')

four_factors_df_losing = compute_four_factors(losing_teams_df, winner=False)
frames = [four_factors_df, four_factors_df_losing]

result = pd.concat(frames, sort=False)
final_df = result.groupby(['Season', 'Team'], as_index=False).agg(

    {'OffensiveShooting': 'mean', 'DefensiveShooting': 'mean', 'OffensiveTOs': 'mean', 'DefensiveTOs': 'mean', 

     'OffensiveRebounding': 'mean', 'DefensiveRebounding': 'mean', 'OffensiveFTs': 'mean', 'DefensiveFTs': 'mean', 'won': 'sum'})

final_df.head()
final_df['ShootingDiff'] = final_df['OffensiveShooting'] - final_df['DefensiveShooting']

final_df.drop(labels=['OffensiveShooting', 'DefensiveShooting'], inplace=True, axis=1)

final_df.head()
final_df['TOsDiff'] = final_df['OffensiveTOs'] - final_df['DefensiveTOs']

final_df.drop(labels=['OffensiveTOs', 'DefensiveTOs'], inplace=True, axis=1)

final_df.head()
final_df['ReboundingDiff'] = final_df['OffensiveRebounding'] - final_df['DefensiveRebounding']

final_df.drop(labels=['OffensiveRebounding', 'DefensiveRebounding'], inplace=True, axis=1)

final_df.head()
final_df['ReboundingFTs'] = final_df['OffensiveFTs'] - final_df['DefensiveFTs']

final_df.drop(labels=['OffensiveFTs', 'DefensiveFTs'], inplace=True, axis=1)

final_df.head()
X = np.array(final_df[['ShootingDiff','TOsDiff', 'ReboundingDiff', 'ReboundingFTs']])
y = np.array(final_df['won'])
reg = LinearRegression().fit(X, y)
reg.score(X, y)
df_sample_sub = pd.read_csv('../input/SampleSubmissionStage1.csv')

n_test_games = len(df_sample_sub)



def get_year_t1_t2(ID):

    """Return a tuple with ints `year`, `team1` and `team2`."""

    return (int(x) for x in ID.split('_'))
X_test_t1 = np.zeros(shape=(n_test_games, 4))

for ii, row in df_sample_sub.iterrows():

    year, t1, t2 = get_year_t1_t2(row.ID)

    row = final_df[(final_df.Team == t1) & (final_df.Season == year)]

    X_test_t1[ii, 0] = row['ShootingDiff']

    X_test_t1[ii, 1] = row['TOsDiff']

    X_test_t1[ii, 2] = row['ReboundingDiff']

    X_test_t1[ii, 3] = row['ReboundingFTs']
X_test_t2 = np.zeros(shape=(n_test_games, 4))

for ii, row in df_sample_sub.iterrows():

    year, t1, t2 = get_year_t1_t2(row.ID)

    row = final_df[(final_df.Team == t2) & (final_df.Season == year)]

    X_test_t2[ii, 0] = row['ShootingDiff']

    X_test_t2[ii, 1] = row['TOsDiff']

    X_test_t2[ii, 2] = row['ReboundingDiff']

    X_test_t2[ii, 3] = row['ReboundingFTs']
t1_preds = reg.predict(X_test_t1)

t1_preds
t2_preds = reg.predict(X_test_t2)

t2_preds
preds = np.zeros(shape=(n_test_games, 1))

for i in range(n_test_games):

    if t1_preds[i] - t2_preds[i] >= 0:

        preds[i] = 0.55

    else:

        preds[i] = 0.45

        

preds
df_sample_sub.Pred = preds

df_sample_sub.head()
df_sample_sub.to_csv('linreg_four_factor.csv', index=False)