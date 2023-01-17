import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns



seeds = pd.read_csv('../input/datafiles/NCAATourneySeeds.csv')

tourney_results = pd.read_csv('../input/datafiles/NCAATourneyCompactResults.csv')

regular_results = pd.read_csv('../input/datafiles/RegularSeasonCompactResults.csv')
def prepare_data(df):

    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']]



    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'

    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'

    df.columns.values[6] = 'location'

    dfswap.columns.values[6] = 'location'         

    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]

    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).sort_index().reset_index(drop=True)

    

    return output
tourney_results = prepare_data(tourney_results)

regular_results = prepare_data(regular_results)
regular_results.head(10)
# convert to str, so the model would treat TeamID them as factors

regular_results['T1_TeamID'] = regular_results['T1_TeamID'].astype(str)

regular_results['T2_TeamID'] = regular_results['T2_TeamID'].astype(str)



# make it a binary task

regular_results['win'] = np.where(regular_results['T1_Score']>regular_results['T2_Score'], 1, 0)



def team_quality(season):

    """

    Calculate team quality for each season seperately. 

    Team strength changes from season to season (students playing change!)

    So pooling everything would be bad approach!

    """

    formula = 'win~-1+T1_TeamID+T2_TeamID'

    glm = sm.GLM.from_formula(formula=formula, 

                              data=regular_results.loc[regular_results.Season==season,:], 

                              family=sm.families.Binomial()).fit()

    

    # extracting parameters from glm

    quality = pd.DataFrame(glm.params).reset_index()

    quality.columns = ['TeamID','beta']

    quality['Season'] = season

    # taking exp due to binomial model being used

    quality['quality'] = np.exp(quality['beta'])

    # only interested in glm parameters with T1_, as T2_ should be mirroring T1_ ones

    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)

    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)

    return quality
team_quality = pd.concat([team_quality(2010),

                          team_quality(2011),

                          team_quality(2012),

                          team_quality(2013),

                          team_quality(2014),

                          team_quality(2015),

                          team_quality(2016),

                          team_quality(2017),

                          team_quality(2018)]).reset_index(drop=True)
sns.set_style('whitegrid')

sns.kdeplot(np.array(team_quality['beta']), bw=0.1)
sns.set_style('whitegrid')

sns.kdeplot(np.array(np.clip(team_quality['quality'],0,1000)), bw=0.1)
team_quality_T1 = team_quality[['TeamID','Season','quality']]

team_quality_T1.columns = ['T1_TeamID','Season','T1_quality']

team_quality_T2 = team_quality[['TeamID','Season','quality']]

team_quality_T2.columns = ['T2_TeamID','Season','T2_quality']



tourney_results['T1_TeamID'] = tourney_results['T1_TeamID'].astype(int)

tourney_results['T2_TeamID'] = tourney_results['T2_TeamID'].astype(int)

tourney_results = tourney_results.merge(team_quality_T1, on = ['T1_TeamID','Season'], how = 'left')

tourney_results = tourney_results.merge(team_quality_T2, on = ['T2_TeamID','Season'], how = 'left')
# we only have tourney results since year 2010

tourney_results = tourney_results.loc[tourney_results['Season'] >= 2010].reset_index(drop=True)



# not interested in pre-selection matches

tourney_results = tourney_results.loc[tourney_results['DayNum'] >= 136].reset_index(drop=True)
seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

seeds['division'] = seeds['Seed'].apply(lambda x: x[0])



seeds_T1 = seeds[['Season','TeamID','seed','division']].copy()

seeds_T2 = seeds[['Season','TeamID','seed','division']].copy()

seeds_T1.columns = ['Season','T1_TeamID','T1_seed','T1_division']

seeds_T2.columns = ['Season','T2_TeamID','T2_seed','T2_division']



tourney_results = tourney_results.merge(seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')

tourney_results = tourney_results.merge(seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
tourney_results['T1_powerrank'] = tourney_results.groupby(['Season','T1_division'])['T1_quality'].rank(method='dense', ascending=False).astype(int)

tourney_results['T2_powerrank'] = tourney_results.groupby(['Season','T2_division'])['T2_quality'].rank(method='dense', ascending=False).astype(int)
piv = pd.pivot_table(tourney_results, index = ['T1_seed'], columns=['T1_powerrank'], values = ['T1_TeamID'], aggfunc=len)

piv = piv.xs('T1_TeamID', axis=1, drop_level=True)
fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(piv, annot=True,cmap='Blues', fmt='g')
tourney_results['win'] = np.where(tourney_results['T1_Score'] > tourney_results['T2_Score'], 1, 0)
mean_win_ratio = pd.DataFrame({'seed_win_ratio': tourney_results.groupby('T1_seed')['win'].mean(),

                               'powerrank_win_ratio': tourney_results.groupby('T1_powerrank')['win'].mean()})

mean_win_ratio
sns.set_style('whitegrid')

sns.lineplot(mean_win_ratio.index, mean_win_ratio['seed_win_ratio']) # Blue

sns.lineplot(mean_win_ratio.index, mean_win_ratio['powerrank_win_ratio']) # Orange
from sklearn.metrics import roc_auc_score



print(f"seed AUC: {roc_auc_score(tourney_results['win'],-tourney_results['T1_seed'])}")

print(f"powerrank AUC: {roc_auc_score(tourney_results['win'],-tourney_results['T1_powerrank'])}")

print(f"team quality AUC: {roc_auc_score(tourney_results['win'],tourney_results['T1_quality'])}")