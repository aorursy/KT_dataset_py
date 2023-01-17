# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
owgr = pd.read_csv('../input/ogwr_historical.csv')
owgr['index'] = owgr.index // 1000
print(owgr.isna().sum())
#Utility functions
def exponential_fit(x, A, B, C):
    return A*np.exp(-B*x)+C

#Function to check if a list decreases monotonically
def check_monotonicity(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def get_bad_values(L, idx):
    bools = [x>=y for x,y in zip(L,L[1:])]
    shift = idx * 1000
    bad_vals = []
    for i in range(len(bools)):
        if not bools[i]:
            bad_vals.append(shift+i)
    return bad_vals
plt.plot(owgr['rank'], owgr['avg_points'], 'o', label='data')
plt.xlabel('Ranking')
plt.ylabel('Average OWGR Points')
plt.plot(owgr['rank'], owgr['avg_points'], 'o', label='data')
plt.axis([50, 300, 0, 4])
plt.xlabel('Ranking')
plt.ylabel('Average OWGR Points')
date_hyp = owgr[owgr['date'] == '05-06-05'] #Get data from the week of 05-06-05
test_hyp = []
for i in range(1, 999):
    ave = (date_hyp['avg_points'].iloc[i-1] + date_hyp['avg_points'].iloc[i+1]) / 2
    test_hyp.append(ave - date_hyp['avg_points'].iloc[i])
    #print(i, ave - date_tmp['avg_points'].iloc[i])
#plt.axis([0,100,-0.5,0.8])
plt.plot(test_hyp)
plt.arrow(x=150, y=-1.0, dx=-60, dy=0.5,
          width=0.05, head_width=0.2, head_length=10, shape='full', color='r')
plt.arrow(x=725, y=0.8, dx=0, dy=-0.5,
          width=0.05, head_width=0.2, head_length=0.1, shape='full', color='r')
# Find all the bad average point values based on if rankings are numerically ordered
bad_vals = []
for i in range(len(owgr.date.unique())):
    date_tmp = owgr[owgr['index'] == i]
    mono = check_monotonicity(date_tmp['avg_points'])
    if not mono:
        bad_vals_slice = get_bad_values(date_tmp['avg_points'], i)
        [bad_vals.append(i) for i in bad_vals_slice]
print(len(bad_vals))
owgr_clean = owgr.drop(owgr.index[bad_vals])
# Compare clean and original data
plt.plot(owgr['rank'], owgr['avg_points'], 'bo', label='original')
plt.plot(owgr_clean['rank'], owgr_clean['avg_points'], 'g.', label='clean')
plt.axis([50, 400, 0, 4])
plt.xlabel('Ranking')
plt.ylabel('Average OWGR Points')
plt.legend()
# Violin plot for distribtuion of OWGR points for each fo top ten players (w/ Tiger)
top_ten = owgr_clean[owgr_clean['rank'] <= 10]
sns.violinplot(x='rank', y='avg_points', data=top_ten, inner=None)
# Violin plot for distribution of OWGR points for each of top ten players (w/o Tiger)
top_ten = owgr_clean[owgr_clean['rank'] <= 10]
top_ten = top_ten[top_ten['name'] != 'TigerWoods']
sns.violinplot(x='rank', y='avg_points', data=top_ten, inner=None)
#Build dataframe with info about players who reached the number 1 ranking
top_players = owgr_clean[owgr_clean['rank'] == 1]['name'].unique()

top_players_dict = {}
for i in top_players:
    player_data = []
    player_data.append(i)
    player_df = owgr_clean[owgr_clean['name'] == i]
    player_data.append(player_df['avg_points'].mean()) #Average points for career
    player_data.append(player_df['rank'].mean())
    player_data.append(player_df.count()['rank']) #Total weeks
    player_data.append(player_df[player_df['rank'] == 1].count()['rank']) #Total weeks at 1
    top_players_dict[i] = player_data
top_players_df = pd.DataFrame.from_dict(top_players_dict, orient='index',
                                        columns=['player', 'ave_points_career', 'ave_rank', 'total_weeks', 'total_weeks_at_1'])
top_players_df['per_weeks_at_1'] = 100 * (top_players_df['total_weeks_at_1'] / top_players_df['total_weeks'])
print(top_players_df.head())
fig, ax = plt.subplots()
ax.scatter(top_players_df['player'], top_players_df['per_weeks_at_1'], 
           color='k', s=30)
ax.set_ylabel('Percent at #1 (%)', color='k')
ax.tick_params(axis='y', labelcolor='k')
plt.xticks(rotation=60)
ax2 =ax.twinx()
ax2.set_ylabel('Average World Ranking', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.scatter(top_players_df['player'], top_players_df['ave_rank'],
            color='b', s=30)
tw = owgr[(owgr['rank'] == 1) & (owgr['name'] == 'TigerWoods')]
notw = owgr[(owgr['rank'] == 1) & (owgr['name'] != 'TigerWoods')]
sns.kdeplot(tw['avg_points'], shade=True, label='Tiger')
sns.kdeplot(notw['avg_points'], shade=True, label='Everyone Else')
plt.xlabel('OWGR Points')
plt.ylabel('Frequency')
plt.title('Average World Ranking Points of #1 Player')
# Tiger vs. Phil vs. Rory over time
tw = owgr_clean[owgr_clean['name'] == 'TigerWoods']
pm = owgr_clean[owgr_clean['name'] == 'PhilMickelson']
rm = owgr_clean[owgr_clean['name'] == 'RoryMcIlroy']

ax = tw.plot(x='index', y='avg_points', color='Red', label='Tiger')
pm.plot(x='index', y='avg_points', color='Green', label='Phil', ax=ax)
rm.plot(x='index', y='avg_points', color='Blue', label='Rory', ax=ax)
plt.ylabel('OWGR Points')
plt.xlabel('Week')