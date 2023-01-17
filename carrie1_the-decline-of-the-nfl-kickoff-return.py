import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import pandas as pd
pd.options.mode.chained_assignment = None

df = pd.read_csv('../input/NFL Play by Play 2009-2018 (v5).csv',low_memory=False)
#let's see what data fields we have
list(df.columns.values)
#what kinds of plays are in the data
df.play_type.unique()
#when was the most recent game recorded
df.sort_values(by='game_date',ascending=False).head()
import seaborn as sns
sns.set_context("talk")
sns.set(palette='bright') 
sns.set(rc={'figure.figsize':(12,6)})
sns.countplot(x="play_type", data=df)
#what data type is the date column and how does it look?
df['game_date'].head()
#make the date field a datetime object
df['date'] =  pd.to_datetime(df['game_date'], format='%Y/%m/%d')
df['date'].head()
import numpy as np

#create year,month and season fields
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df['season'] = np.where(df['month']<8, df['year']-1,df['year'])

#play types by season
sns.set(rc={'figure.figsize':(16,4)})
sns.countplot(x="play_type", hue="season", data=df)
df['kickoff'] = np.where(df.play_type =='kickoff',1,0)
df.kickoff.sum() / df.kickoff.count() * 100
df_season = df.groupby(by=['season'])
df_season.kickoff.sum() / df_season.kickoff.count() * 100
#make a dataset with only kickoffs and see how many were returned by season
kickoffs = df[df.play_type == 'kickoff']
kickoffs['percent_returned'] = np.where(kickoffs.return_yards >0,1,0)
kickoffs_season = kickoffs.groupby(by=['season'])
temp = kickoffs_season[['percent_returned']].mean()
temp = round(temp,2)
sns.set(rc={'figure.figsize':(8,10)})
sns.set_context("poster")
ax=sns.heatmap(temp, annot=True, linewidths=.5, cbar=False,square=False,cmap="YlGnBu")