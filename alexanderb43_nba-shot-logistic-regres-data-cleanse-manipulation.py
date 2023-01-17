import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

%matplotlib inline
df = pd.read_csv("../input/NBA.csv")
df.columns
df.dtypes
df.drop(['FINAL_MARGIN'], axis = 1, inplace = True)
df['MATCHUP'] = df['MATCHUP'].str[:12]
df['MATCHUP'] = pd.to_datetime(df.MATCHUP)
df['Day'] = df.MATCHUP.dt.day

df['Month'] = df.MATCHUP.dt.month

df['Year'] = df.MATCHUP.dt.year
df['GAME_CLOCK'] = pd.to_datetime(df.GAME_CLOCK, format = '%M:%S')
df['GAME_CLOCK'] = df.GAME_CLOCK.dt.time
df.dropna(subset = ['SHOT_CLOCK'], how = 'any', inplace = True)
df['SHOT_CLOCK'] = pd.to_datetime(df.SHOT_CLOCK, format = '%S')
df['SHOT_CLOCK'] = df.SHOT_CLOCK.dt.time
df['LOCATION'] = df['LOCATION'].str.replace('H','0').str.replace('A','1')
df['LOCATION'] = df['LOCATION'].astype(int)
df['SHOT_RESULT'] = df['SHOT_RESULT'].str.replace('missed','0').str.replace('made','1').astype(int)

df.drop('FGM',axis = 1, inplace = True)
df.head()
df.groupby(['player_name']).SHOT_RESULT.sum().sort_values(ascending = False).head(5)
df.groupby(['player_name']).PTS.sum().sort_values(ascending = False).head(5)
df.groupby(['CLOSEST_DEFENDER']).CLOSE_DEF_DIST.mean().sort_values().head(50)
df.MATCHUP.min()
df.MATCHUP.max()
df['MY'] = df.Year.map(str) + '-' + df.Month.map(str)
df.groupby(['Month','Year','player_name']).PTS.sum().sort_values(ascending = False).head(5)
df.groupby(['player_name']).SHOT_DIST.mean().sort_values(ascending = False).head(5)
df.groupby(['Month']).PTS.sum()
df.dtypes
df.groupby(['player_name']).PTS.sum().sort_values(ascending = False).head(10).plot.bar()

plt.title('Most Points during October 2014 to March 2015')

plt.xlabel('Player')

plt.xticks(color = 'B')

plt.ylabel('Points Scored')
df.groupby(['MY']).PTS.sum().plot(c = 'r')

plt.title('Point Total by Month')

plt.xlabel('Year-Month')

plt.ylabel('Points')

plt.yticks([5000,10000,15000,20000,25000,30000,35000],['5,000',

                                                      '10,000','15,000','20,000','25,000','30,000','35,000'])
df.corr()
plt.scatter(x=df.CLOSE_DEF_DIST,y=df.SHOT_DIST, c = 'b')

plt.title('Defender Distance Vs Shot Distance')

plt.xlabel('Defender Distance')

plt.ylabel('Shot Distance')

plt.scatter(x=df.TOUCH_TIME,y=df.SHOT_DIST, c = 'b')

plt.title('Touch Time Vs Shot Distance')

plt.xlabel('Touch Time')

plt.ylabel('Shot Distance')
df[df.TOUCH_TIME <-80]
df.drop([5574], axis = 0, inplace = True)
plt.scatter(x=df.TOUCH_TIME,y=df.SHOT_DIST, c = 'b')

plt.title('Touch Time Vs Shot Distance')

plt.xlabel('Touch Time')

plt.ylabel('Shot Distance')
df[df.TOUCH_TIME <-0].head()
df.drop(df[df.TOUCH_TIME <-0].index, inplace = True)
plt.scatter(x=df.TOUCH_TIME,y=df.SHOT_DIST, c = 'b')

plt.title('Touch Time Vs Shot Distance')

plt.xlabel('Touch Time')

plt.ylabel('Shot Distance')
X = df.ix[:,[9,15]]

Y = df.SHOT_DIST

x = sm.add_constant(X)
Reg = sm.OLS(Y,x)

FitReg = Reg.fit()

FitReg.summary()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df[['SHOT_DIST','CLOSE_DEF_DIST','TOUCH_TIME','DRIBBLES']],df.SHOT_RESULT,test_size = 0.3)
model = LogisticRegression()

model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)
model.predict_proba(X_test)
df.columns
df.head(1)
df.SHOT_RESULT