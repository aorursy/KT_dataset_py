%matplotlib inline



import pandas as pd

import numpy as np

import warnings

import pandas as pd

import seaborn as sns

import math

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display, HTML

from astropy.table import Table

from scipy import stats

from statsmodels.formula.api import ols

warnings.simplefilter(action = "ignore")



all_plays = pd.read_csv('../input/nflplaybyplay2015.csv')

play_codes = {'Punt': 'Punt', 'Field Goal': 'Field Goal', 'Pass': 'Play Attempt (Pass/Run/Sack)', 'No Play': 'Other', 'Run': 'Play Attempt (Pass/Run/Sack)', 'Sack': 'Play Attempt (Pass/Run/Sack)', 'QB Kneel': 'Other', 'Timeout': 'Other'}

all_plays['Play'] = all_plays['PlayType'].map(play_codes)



short_flag = all_plays['ydstogo'] < 5

med_flag = (all_plays['ydstogo'] >= 5) & (all_plays['ydstogo'] <= 10)

long_flag = all_plays['ydstogo'] > 10



all_plays.loc[short_flag, 'dist_cat'] = 'Short'

all_plays.loc[med_flag, 'dist_cat'] = 'Medium'

all_plays.loc[long_flag, 'dist_cat'] = 'Long'

other_plays = all_plays[all_plays['down'] != 4]

fourth_plays = all_plays[all_plays['down'] == 4]

fourth_plays = fourth_plays.drop(['Unnamed: 0', 'PlayAttempted', 'Season'], 1)

converted_flag = fourth_plays['ydstogo'] > fourth_plays['Yards.Gained']

fourth_plays['converted'] = 1

fourth_plays.loc[converted_flag, 'converted'] = 0



attempts = fourth_plays[fourth_plays['Play'] == 'Play Attempt (Pass/Run/Sack)']

punts = fourth_plays[fourth_plays['Play'] == 'Punt']

field_goals = fourth_plays[fourth_plays['Play'] == 'Field Goal']

other_plays = fourth_plays[fourth_plays['Play'] == 'Other']



first_qtr = fourth_plays[fourth_plays['qtr'] == 1]

second_qtr = fourth_plays[fourth_plays['qtr'] == 2]

third_qtr = fourth_plays[fourth_plays['qtr'] == 3]

fourth_qtr = fourth_plays[fourth_plays['qtr'] == 4]
att = attempts['yrdline100']

punt = punts['yrdline100']

fg = field_goals['yrdline100']

other = other_plays['yrdline100']

rows = ['4th Down Attempt', 'Punt', 'FG', 'Other']

    

distance_avg = [round(np.mean(att), 2), round(np.mean(punt), 2), round(np.mean(fg), 2), round(np.mean(other), 2)]

distance_mode = [stats.mode(att).mode, stats.mode(punt).mode, stats.mode(fg).mode, stats.mode(other).mode]

distance_sd = [round(np.std(att), 2), round(np.std(punt), 2), round(np.std(fg), 2), round(np.std(other), 2)]    

tbl = Table([rows, distance_avg, distance_mode, distance_sd], names=('Type', 'Average Dist to Goal', 'Mode', 'SD'))

print(tbl)
g1 = sns.FacetGrid(fourth_plays, col='Play')

g1.map(plt.hist, 'yrdline100')

g2 = sns.FacetGrid(fourth_plays, col='Play')

g2.map(sns.boxplot, 'yrdline100')

sns.plt.show()
g1 = sns.FacetGrid(fourth_plays, col='Play')

g1.map(plt.hist, 'TimeSecs')



g1 = sns.FacetGrid(fourth_plays, col='Play')

g1.map(sns.boxplot, 'TimeSecs')

sns.plt.show()
g1 = sns.FacetGrid(fourth_plays, col='Play')

g1.map(plt.hist, 'ydstogo')



g1 = sns.FacetGrid(fourth_plays, col='Play')

g1.map(sns.boxplot, 'ydstogo')

sns.plt.show()
standings15 = pd.read_csv('../input/standings2015.csv')

fourth_plays_vc = dict(fourth_plays['posteam'].value_counts())

attempts_vc = dict(attempts['posteam'].value_counts())

standings15['4thPossesions'] = standings15['Team'].map(fourth_plays_vc)

standings15['4thAttempts'] = standings15['Team'].map(attempts_vc)

standings15['PercentAttempts'] = standings15['4thAttempts']/standings15['4thPossesions']

sns.regplot(standings15['PercentAttempts'], standings15['Win'])

plt.title('Percent of 4th Down Attempts vs. Wins')
display(standings15)
model = ols('Win~PercentAttempts', standings15).fit()

model.summary()
g1 = sns.FacetGrid(fourth_plays, col='Play')

bins_fg = [-27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

bins_td = [-28, -21, -14, -7, 0, 7, 14, 21, 28]

g1.map(sns.distplot, 'ScoreDiff', bins=bins_td)

g1 = sns.distplot(attempts['ScoreDiff'], bins=bins_td)

plt.title('Distribution of Score Difference on 4th Down Attempts')

g1 = sns.distplot(punts['ScoreDiff'], bins=bins_td)

plt.title('Distribution of Score Difference on Punts')

plt.draw()

run_plays_4D = len(fourth_plays['PlayType'] == 'Run')

all_runs = len(all_plays['PlayType'] == 'Run')

pass_plays_4D = len(fourth_plays['PlayType'] == 'Pass') + len(attempts['PlayType'] == 'Sack')

all_pass = len(all_plays['PlayType'] == 'Pass') + len(all_plays['PlayType'] == 'Sack')



play_usage = ['Run', 'Pass']



print('Total Plays:', len(all_plays))

print('Total 4D Plays:', len(fourth_plays), '\n')

runs_and_passes = {

                    '4th Down': pd.Series([run_plays_4D, pass_plays_4D], index=play_usage), 

                    '4th Down Ratio': pd.Series([run_plays_4D/len(all_plays), pass_plays_4D/len(all_plays)], index=play_usage),

                    'All Downs': pd.Series([all_runs, all_pass], index=play_usage), 

                    'Percentage of All Plays': pd.Series([all_runs/len(all_plays), all_pass/len(all_plays)], index=play_usage)}

display(pd.DataFrame(runs_and_passes))