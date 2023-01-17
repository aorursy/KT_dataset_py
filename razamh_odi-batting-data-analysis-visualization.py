# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
odi = pd.read_csv('/kaggle/input/odi-batting.csv')

odi.head()
odi.info()
odi["MatchDate"].head()
# odi['MatchDate'] = pd.to_datetime(odi['MatchDate'], format="%M-%d-%Y")
odi["MatchDate"].head()
odi.info()
odi.describe()
#25%: 25% values less than number displayed

odi.describe(include='object')
odi.describe(include='all') #all column summary
odi['Player']
odi['Player'].head()
odi[["Player","Runs"]].head(10)
odi[['Player','Runs','MatchDate']].head(10)
#top ten players based on total number of matches:

odi['Player'].value_counts().head(10)
print("Mean of the Runs Column is ",odi['Runs'].mean())

print("Median of the Runs Column is ",odi['Runs'].median())

print("Maximum runs of the Runs Column is ",odi['Runs'].max())

print("Minimum Runs in Runs Column is ",odi['Runs'].min())
#one column one value filter

sachin_rows = odi[(odi['Player']=='Sachin R Tendulkar')]

sachin_rows #index is not altered. if this was the index in the original file(excel in this case), it remains the same here
sachin_rows = odi[(odi['Player']=='Sachin R Tendulkar')].head()

sachin_rows
sachin = odi[(odi['Player']=='Sachin R Tendulkar')].shape

sachin
#DROP rows and columns

if 'URL' in odi.columns:

    odi = odi.drop(['URL'], axis=1)

odi.shape  #one column reduced. earlier, we had 8
odi.head()
players_summary = odi.groupby(['Player'])

len(players_summary)
players_summary = odi.groupby(['Player']).agg({'Runs':'sum'})

players_summary
players_summary.loc['Sachin R Tendulkar','Runs']
#top ten players

players_summary.sort_values(['Runs'],ascending=False).head(10)
odi['is_century'] = odi['Runs'].apply(lambda run:1 if run>99 else 0)

odi['is_fifty'] = odi['Runs'].apply(lambda run:1 if run>49 and run<100 else 0)

odi['is_duck'] = odi['Runs'].apply(lambda run:1 if run==0 else 0)

odi['missed_century'] = odi['Runs'].apply(lambda run:1 if run>90 and run<100 else 0)

odi.head()
players_summary = odi.groupby(['Player']).agg({'Runs':'sum','is_century':'sum','is_fifty':'sum','is_duck':'sum',

                                               'missed_century':'sum','ScoreRate':'mean','Country':'count'})

players_summary = players_summary.rename(columns = {'Country':'No. of matches'})

players_summary.sort_values(['Runs'],ascending=False).head(10)
odi.info()
odi['year'] = odi['MatchDate'].apply(lambda d: d[-4:])

odi.head()
sachin_rows = odi[(odi['Player'] == 'Sachin R Tendulkar')]

sachin_years = sachin_rows.groupby(['year']).agg({'Runs':'sum'})

sachin_years
odi['score1'] = odi['ScoreRate'].apply(lambda d: d>100)

odi.head()
#rahul_rows = odi[odi['Player'].strcontains('Rahul')] #another way

rahul_rows = odi[odi['Player'] == 'Rahul Dravid']

rahul_years = rahul_rows.groupby(['year']).agg({'Runs':'sum'})

rahul_years
df_new = pd.concat([sachin_years,rahul_years])

df_new
df_new = pd.concat([sachin_years,rahul_years],axis=1,sort=False, join='inner')

df_new
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
odi.head()
players = ['Sachin R Tendulkar','Rahul Dravid','Virender Sehwag']

odi1 = odi[(odi['Player'].isin(players))]

summary1 = odi1.groupby(['Player','Versus']).agg({'Runs':'mean'})

summary1

wide = summary1.unstack(level=1)

sns.heatmap(wide,cmap='Blues')

plt.show()
summary_pivot = pd.pivot_table(odi1,index='Player',columns='Versus',values='Runs',aggfunc='mean')   #index=Player, columns=Versus

summary_pivot
sns.heatmap(summary_pivot,cmap='Blues')

plt.show()
top_players = odi['Player'].value_counts().head(10)

top_players
#x axis player names

# yaxis player numbers

top_players.index
xvalues = top_players.index

yvalues = top_players.values

plt.figure(figsize=(15,10))

plt.bar(xvalues,yvalues)

plt.xticks(rotation=30)

plt.xlabel('Player')

plt.ylabel('Total Matches')

plt.title('top ten players by total no. of matches')

plt.show()
# Using pandas

top_players.plot.bar()

plt.show()
odi['year'] = odi['MatchDate'].str[-4:]

odi['year']

sachin_rows = odi[odi['Player'].str.contains('Sachin')]

years_runs = sachin_rows.groupby(['year'])['Runs'].sum()

#years_runs = sachin_rows.groupby(['year']).agg({'Runs':'sum'})['Runs']

years_runs.plot.line()

plt.show()
player = odi['Player'].value_counts().head()

plt.figure()

player.plot.pie()

plt.ylabel('')

plt.title('Player Name')

plt.get_cmap()

plt.show()
sns.barplot(data=odi1,x='Player',y='Runs')
sns.pointplot(data=odi, orient='h')
Complete_Data = odi.to_csv("Updata ODi Batting", index = False)