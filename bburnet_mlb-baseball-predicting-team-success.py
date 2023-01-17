import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# import libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

# import data
df_b = pd.read_csv('/kaggle/input/Batting.csv')
df_f = pd.read_csv('/kaggle/input/Fielding.csv')
df_p = pd.read_csv('/kaggle/input/Pitching.csv')
df_t = pd.read_csv('/kaggle/input/Teams.csv')
df_n = pd.read_csv('/kaggle/input/People.csv')
df_b.columns
df_f.columns
df_p.columns
df_t.columns
df_n.columns
### filtering out conflicting/unnecessary features
df_b = df_b.drop(['stint', 'lgID'], axis = 1)
df_f = df_f.drop(['stint', 'lgID', 'PB', 'WP', 'SB', 'CS', 'ZR'], axis = 1)
df_p = df_p.drop(['stint', 'lgID'], axis = 1)
df_n = df_n[['playerID', 'birthYear', 'nameFirst', 'nameLast', 'weight', 'height', 'bats', 'throws']]
df_t = df_t[['yearID', 'lgID', 'teamID', 'franchID', 'divID', 'Rank', 'W', 'L','DivWin', 'WCWin', 'LgWin', 'WSWin']]

# rename columns to avoid conflicting with pitcher W/L columns when merging
df_t.rename(columns = {'W':'team_wins'}, inplace = True)
df_t.rename(columns = {'L':'team_loses'}, inplace = True)
df_merge_b_n = pd.merge(df_b, df_n, on = 'playerID')
df_merge_f_n = pd.merge(df_f, df_n, on = 'playerID')
df_merge_p_n = pd.merge(df_p, df_n, on = 'playerID')
df_bat = pd.merge(df_merge_b_n, df_t, on = ['teamID', 'yearID'])
df_field = pd.merge(df_merge_f_n, df_t, on = ['teamID', 'yearID'])
df_pitch = pd.merge(df_merge_p_n, df_t, on = ['teamID', 'yearID'])
# filtering by year
df_bat = df_bat[(df_bat.yearID >= 1980) & (df_bat.yearID != 1994)]
df_field = df_field[(df_field.yearID >= 1980) & (df_field.yearID != 1994)]
df_pitch = df_pitch[(df_pitch.yearID >= 1980) & (df_pitch.yearID != 1994)]
df_bat_STL_2018 = df_bat[(df_bat.teamID == 'SLN') & (df_bat.yearID == 2018)]
df_bat_STL_2018.head(20)
   
df_field_STL_2018 = df_field[(df_field.teamID == 'SLN') & (df_field.yearID == 2018)]
df_field_STL_2018.head(20)

df_pitch_STL_2018 = df_pitch[(df_pitch.teamID == 'SLN') & (df_pitch.yearID == 2018)]
df_pitch_STL_2018.head(20)
# filtering by at bats - we don't want to account for those who barely contributed to a team, also a reasonable sample size
df_bat['AB'].describe()
df_bat['AB'].hist(bins = 50)
df_bat = df_bat[df_bat['AB'] >= 250]
#filtering by innings fielded - for the same reason, also getting rid of pitchers from this table
df_field['InnOuts'].describe()
df_field['InnOuts'].hist(bins = 50)
df_field = df_field[df_field['POS'] != 'P']
df_field = df_field[df_field['InnOuts'] >= 1000]
# filtering by innings pitched - for the same reason
df_pitch['IPouts'].describe()
df_pitch['IPouts'].hist(bins = 50)
df_pitch = df_pitch[df_pitch['IPouts'] >= 200]
### forming a universal batting statistic - wOBA ###
# data set does not account for singles, let's create that... it's necessary for calculating wOBA
df_bat['1B'] = df_bat['H'] - (df_bat['2B'] + df_bat['3B'] + df_bat['HR'])                                  

#creating wOBA
df_bat['wOBA'] = ((0.69 * df_bat['BB']) + (0.72 * df_bat['HBP']) + (0.89 * df_bat['1B'])\
               + (1.27 * df_bat['2B']) + (1.62 * df_bat['3B']) + (2.10 * df_bat['HR']))\
    / (df_bat['AB'] + df_bat['BB'] - df_bat['IBB'] + df_bat['SF'] + df_bat['HBP'])
### creating team averages - using groupby function ###
# Team_wOBA
df_bat['Team_wOBA'] = df_bat.groupby(['teamID', 'yearID']).wOBA.transform('mean')

# Team_E
df_field['Team_E'] = df_field.groupby(['teamID', 'yearID']).E.transform('mean')

# Team_ERA
df_pitch['Team_ERA'] = df_pitch.groupby(['teamID', 'yearID']).ERA.transform('mean')
# import libaries
import seaborn as sns
import matplotlib.pyplot as plt

import scipy
from scipy import stats
# wOBA and Rank
ax = sns.jointplot(
    x = "Rank",
    y = "wOBA",
    data = df_bat,
    kind = "kde",
    height = 11)
ax.ax_joint.set_xlabel('Team Rank', fontsize = 16, fontweight = 'bold')
ax.ax_joint.set_ylabel('wOBA', fontsize = 16, fontweight = 'bold')

plt.show()

pc1 = scipy.stats.pearsonr(df_bat['wOBA'], df_bat['Rank']) 
print('wOBA and Team Rank: correlation coefficient = ', round(pc1[0],2), ', p-value = ', round(pc1[1],4))
# Errors and Rank
ax = sns.jointplot(
    x = "Rank",
    y = "E",
    data = df_field,
    kind = "kde",
    height = 11)
ax.ax_joint.set_xlabel('Team Rank', fontsize = 16, fontweight = 'bold')
ax.ax_joint.set_ylabel('Errors', fontsize = 16, fontweight = 'bold')

plt.show()

pc2 = scipy.stats.pearsonr(df_field['E'], df_field['Rank'])
print('Errors and Team Rank: correlation coefficient = ', round(pc2[0],2), ', p-value = ', round(pc2[1],4))
# ERA and Rank
ax = sns.jointplot(
    x = "Rank",
    y = "ERA",
    data = df_pitch,
    kind = "kde",
    height = 11)
ax.ax_joint.set_xlabel('Team Rank', fontsize = 16, fontweight = 'bold')
ax.ax_joint.set_ylabel('ERA', fontsize = 16, fontweight = 'bold')

plt.show()

pc3 = scipy.stats.pearsonr(df_pitch['ERA'], df_pitch['Rank'])
print('ERA and Team Rank: correlation coefficient = ', round(pc3[0],2), ', p-value = ', round(pc3[1],4))
# team_wOBA and Rank
ax = sns.jointplot(
    x = "Rank",
    y = "Team_wOBA",
    data = df_bat,
    kind = "kde",
    height = 11)
ax.ax_joint.set_xlabel('Team Rank', fontsize = 16, fontweight = 'bold')
ax.ax_joint.set_ylabel('Team wOBA', fontsize = 16, fontweight = 'bold')

plt.show()

pc4 = scipy.stats.pearsonr(df_bat['Team_wOBA'], df_bat['Rank'])
print('Team wOBA and Team Rank: correlation coefficient = ', round(pc4[0],2), ', p-value = ', round(pc4[1],4))
# team_E and Rank
ax = sns.jointplot(
    x = "Rank",
    y = "Team_E",
    data = df_field,
    kind = "kde",
    height = 11)
ax.ax_joint.set_xlabel('Team Rank', fontsize = 16, fontweight = 'bold')
ax.ax_joint.set_ylabel('Team Errors', fontsize = 16, fontweight = 'bold')

plt.show()

pc5 = scipy.stats.pearsonr(df_field['Team_E'], df_field['Rank'])
print('Team Errors and Team Rank: correlation coefficient = ', round(pc5[0],2), ', p-value = ', round(pc5[1],4))
# team_ERA and Rank
ax = sns.jointplot(
    x = "Rank",
    y = "Team_ERA",
    data = df_pitch,
    kind = "kde",
    height = 11)
ax.ax_joint.set_xlabel('Team Rank', fontsize = 16, fontweight = 'bold')
ax.ax_joint.set_ylabel('Team ERA', fontsize = 16, fontweight = 'bold')

plt.show()

pc6 = scipy.stats.pearsonr(df_pitch['Team_ERA'], df_pitch['Rank'])
print('Team ERA and Team Rank: correlation coefficient = ', round(pc6[0],2), ', p-value = ', round(pc6[1],4))