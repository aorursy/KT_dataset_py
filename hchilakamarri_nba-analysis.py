
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
## NBA Team Statistics from 17/18 Season

nba_team = pd.read_csv('../input/nbateam-stats-1718/18')
nba_team = nba_team.sort_values('Team')
nba_team['W%'] = ''

## NBA Misc. Team Statistics from 17/18 Season, includes Wins and Losses

nba_misc = pd.read_csv('../input/nba-mic-stats-1718/18.csv')
nba_misc = nba_misc.sort_values('Team')
nba_team['W%'] = (nba_misc['W']/82).round(3)
nba_team['Pace'] = nba_misc['Pace']

#Drop League Average
nba_team = nba_team.set_index("Team")
nba_team = nba_team.drop("League Average", axis=0)

#nba_misc.head()
nba_team.describe()

#Plot 
plt.scatter(nba_team['PTS'].round(2), nba_team['W%'])
plt.axis([98, 114, 0, .85])
plt.xlabel('Points Per Game')
plt.ylabel('Win %')

d = {'PPG': nba_team['PTS'], 'W%': nba_team['W%']}
df = pd.DataFrame(data=d)


sns.lmplot(x='PPG', y='W%', data=df, aspect=1.5, scatter_kws={'alpha':0.2})
# fit a linear regression model based on PPG to calculate Win %
from sklearn.linear_model import LinearRegression
model = LinearRegression()

X = df[['PPG']]
y = df['W%']

#np.any(np.isnan(X)) #False
#np.all(np.isfinite(X)) #True

#np.any(np.isnan(y)) #False
#np.all(np.isfinite(y)) #True

model.fit(X, y)   #:D
df['pred'] = model.predict(X)

# put the plots together
plt.scatter(df['PPG'], df['W%'])
plt.plot(df['PPG'], df['pred'], color='red')
plt.xlabel('Points Per Game')
plt.ylabel('Win %')
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, df['pred'])
#Taking into account a teams pace
#Pace estimates number of posessions per game

nba_team['PPP'] = (nba_team['PTS'] / nba_team['Pace'])
plt.scatter(nba_team['PPP'], nba_team['W%'])
#plt.axis([.95, 1.175, 0, .85])


model2 = LinearRegression()
X2 = nba_team[['PPP']]
y2 = nba_team['W%']
model2.fit(X2, y2)

PPP_pred = model2.predict(X2)

plt.plot(X2, PPP_pred , color='red')

print(mean_absolute_error(y2, PPP_pred))
#use entire table to estimate win percentage
X3 = nba_team[['FG%', '3P%', 'DRB', 'AST','TOV', 'PTS']]
y3 = nba_team['W%']

model3 = LinearRegression()
model3.fit(X3, y3)

## creating model using more variables
d3 = {'FG%': nba_team['FG%'], '3P%': nba_team['3P%'], 'DRB': nba_team['DRB'], 'AST': nba_team['AST'], 'TOV': nba_team['TOV'], 'PTS': nba_team['PTS'], 'W%': nba_team['W%']}
df3 = pd.DataFrame(data=d3)
df3['pred'] = model3.predict(X3)
#print(df3)
mean_absolute_error(y3, df3['pred'])
## Find out if FG% is normalized by Points
nba_ind = pd.read_csv('../input/nba-player-totals-1718/18.csv')
nba_ind = nba_ind.dropna()

nba_ind.describe()

plt.scatter(nba_ind['FG%'], nba_ind['PTS'])
plt.xlabel('Field Goal %')
plt.ylabel('Points')
plt.scatter(nba_ind['3P%'], nba_ind['3P']*3)
plt.xlabel('3-Point %')
plt.ylabel('Total Points')
plt.scatter(nba_ind['2P%'], nba_ind['2P']*2)
plt.xlabel('2-Point %')
plt.ylabel('Total Points')

mean_2PT = np.mean(nba_ind['2P%'])
mean_FG = np.mean(nba_ind['FG%'])
nba_ind['Diff_Mean_FG'] = abs(nba_ind['FG%'] - mean_FG)
#print(nba_ind['Diff_Mean_FG'])

#plot absolute value of difference to the mean vs total points scored

plt.scatter(nba_ind['Diff_Mean_FG'], nba_ind['PTS'])
plt.xlabel('Absolute Difference from Mean')
plt.ylabel('Points')

X3 = nba_ind[['Diff_Mean_FG']]
y3 = nba_ind['PTS']

model3 = LinearRegression()
model3.fit(X3, y3)
fgpct_predict = model3.predict(X3)

plt.plot(nba_ind['Diff_Mean_FG'], fgpct_predict, color='red')
#standardized NBA Individual Stats
#filter out players who played less than 300 minutes 
nba_std = nba_ind[nba_ind.MP >= 300]

#Points from field goals per minute versus fg%
nba_std['FGPTS'] = (nba_std['3P'] * 3) + (nba_std['2P'] * 2)

#Points per minute from field goals
nba_std['PTS/MP'] = nba_std['FGPTS'] / nba_std['MP']

nba_std.describe()
plt.scatter(nba_std['FG%'], nba_std['PTS/MP'])
plt.xlabel('Field Goal %')
plt.ylabel('Points from FG per Minute Played')
nba_18 = nba_team
nba_18['Season'] = '17-18'

nba_10 = pd.read_csv('../input/nba-team-stats-0910/10.csv')
nba_10['Season'] = '09-10'

nba_04 = pd.read_csv('../input/nba-team-stats-0304/04.csv')
nba_04['Season'] = '03-04'
nba_04.head()

##FGA
sns.distplot(nba_04['FGA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_10['FGA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_18['FGA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
##FGA
sns.distplot(nba_04['FG%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_10['FG%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_18['FG%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
##need to add legends
# FG%
sns.distplot(nba_04['2P%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_10['2P%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_18['2P%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
##FGA
sns.distplot(nba_04['2PA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_10['2PA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_18['2PA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

##3pt %
sns.distplot(nba_04['3P%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
sns.distplot(nba_10['3P%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
sns.distplot(nba_18['3P%'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
##3pt Attempts
sns.distplot(nba_04['3PA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_10['3PA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})

sns.distplot(nba_18['3PA'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})