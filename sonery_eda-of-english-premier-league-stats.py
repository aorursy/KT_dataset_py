import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

%matplotlib inline
df_epl = pd.read_csv("../input/epl-stats-20192020/epl2020.csv")

print(df_epl.shape)
pd.set_option("display.max_columns",45)

df_epl.head()
#Drop redundant feature column
df_epl.drop(['Unnamed: 0'], axis=1, inplace=True)

#reset the index
df_epl = df_epl.reset_index(drop=True)
df_epl.columns
df_epl.matchDay.value_counts()
df_epl[['teamId','tot_points']].groupby('teamId').max().sort_values(by='tot_points', ascending=False)[:10]
plt.figure(figsize=(10,6))
plt.title("Expected vs Actual Goals - Distribution of Difference", fontsize=18)

diff_goal = df_epl.xG - df_epl.scored

sns.distplot(diff_goal, hist=False, color='blue')
df_epl[df_epl.h_a == 'h'][['xG','HS.x','HST.x','HtrgPerc','tot_goal']].corr()
df_epl[df_epl.h_a == 'a'][['xG','AS.x','AST.x','AtrgPerc','tot_goal']].corr()
df_epl['keep_performance'] = df_epl['missed'] / df_epl['xGA']
df_epl[['teamId','keep_performance']].groupby('teamId').mean().sort_values(by='keep_performance', ascending=False)
plt.figure(figsize=(10,6))
plt.title("Expected vs Actual Points - Distribution of Difference", fontsize=18)

diff_pts = df_epl.xpts - df_epl.pts

sns.distplot(diff_pts, hist=False, color='blue')
df_epl[df_epl.teamId == 'Man City'][['pts','matchDay']].groupby('matchDay').agg(['mean','count'])
df_epl['goals']= df_epl['scored'] + df_epl['missed']
df_epl['goals'].mean()
df_epl[['h_a','scored','pts']].groupby('h_a').mean()
print("Home team stats \n {} \n".format(df_epl[df_epl.h_a == 'h'][['HS.x','HST.x','HtrgPerc']].mean()))
print("Away team stats \n {} \n".format(df_epl[df_epl.h_a == 'a'][['AS.x','AST.x','AtrgPerc']].mean()))
df_epl['performance'] = df_epl['pts'] - df_epl['xpts']
df_perf = df_epl[['teamId','performance']].groupby('teamId').mean().sort_values(by='performance', ascending=False)
    
print("Above expectation \n {} \n".format(df_perf[df_perf.performance > 0]))
print("Below expectation \n {} \n".format(df_perf[df_perf.performance < 0]))
df_epl['cards'] = df_epl['HY.x'] + df_epl['HR.x'] + df_epl['AY.x'] + df_epl['AR.x']
df_epl[['Referee.x','cards']].groupby('Referee.x').agg(['mean','count'])
liv = df_epl[df_epl.teamId == 'Liverpool']
liv.shape
print("Home shots \n {} \n".format(liv[liv.h_a == 'h'][['HS.x','HST.x','HtrgPerc']].mean()))
print("Away shots \n {} \n".format(liv[liv.h_a == 'a'][['AS.x','AST.x','AtrgPerc']].mean()))