
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
plt.title('Win Percentage vs. PPG in 2017/18 Season')
plt.axhline(y=0.5, color='green', linestyle=':')

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
plt.axhline(y=0.5, color='green', linestyle=':')
plt.title('Predictive Model: Win Percentage vs. PPG in 2017/18 Season')
plt.xlabel('Points Per Game')
plt.ylabel('Win %')
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, df['pred'])
league_avg = pd.read_csv('../input/league-averages-20002018/League Averages.csv') 
league_avg = league_avg.reindex(index=league_avg.index[::-1])
league_avg.head()
plt.bar(league_avg['Season'], league_avg['PTS'])
plt.ylim(90,110)
plt.xticks(league_avg['Season'], color='black', rotation=45, fontsize='10', horizontalalignment='right')
plt.ylabel('Points Per Game')
plt.title('Average PPG Since 00/01 Season')
#FG%
plt.plot(league_avg['Season'], league_avg['FG%'],  '-o', alpha = 1)
#3P%
plt.plot(league_avg['Season'], league_avg['3P%'],'-o', alpha = 1)
plt.xticks(league_avg['Season'], color='black', rotation=45, fontsize='10', horizontalalignment='right')
plt.ylim(0.00, 1)
plt.ylabel('Shooting %')
plt.title('Average Shooting Pct Since 00/01 Season')
legend = [league_avg['FG%'], league_avg['3P%']]
plt.legend()
#FGA
plt.plot(league_avg['Season'], league_avg['FGA'],'-o',alpha = 1)
plt.ylim(65,100)
plt.xticks(league_avg['Season'], color='black', rotation=45, fontsize='10', horizontalalignment='right')
plt.ylabel('Field Goal Attempts')
plt.title("Average FGA Since '00/01 Season")
plt.plot(league_avg['Season'], league_avg['3PA'], '-o', alpha = 1, color = 'orange')
plt.ylim(0,35)
plt.xticks(league_avg['Season'], color='black', rotation=45, fontsize='10', horizontalalignment='right')
plt.ylabel('3-Point Attempts')
plt.title("Average 3PA since '00/01")