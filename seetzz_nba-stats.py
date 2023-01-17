import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

champion = pd.read_csv('../input/championsdata.csv')
champion.head()
champion.shape
len(champion)
champion.columns
champion.size
champion[['FGP','TPP','FTP']].round(2).head()
champion.isnull().sum().sum()
champion.isnull().sum().idxmax()
champion['Team'].unique().tolist()
champion.groupby(['Year','Team','Game', 'Home'], as_index=True).agg({"FG": "sum", "FGA": "sum","Win": "max", "PTS": "sum"})
champion.groupby('Team')[['Game','Win','PTS']].sum()
champion.groupby('Year')['Win'].count().sum()
year_wins = pd.crosstab(champion.Win, champion.Year, margins=True)

year_wins.T
dfhome_1 = champion[champion['Home'] == 1 ]

dfhome_1.head()
champion.groupby('Team')['Win'].agg(np.sum).plot(kind = 'bar')
over = champion.groupby('Team', as_index=True).agg({"PTS": "sum"})

over['PTS'].plot(kind='bar')
champion[(champion['Home']>0) & (champion['Year'] == 2000) & (champion['Team'] == 'Lakers')]
champion[(champion['Home']>0) & (champion['Team'] == 'Lakers') | (champion['Team'] == 'Bulls')]
champion[champion["Team"] == "Lakers"]["PTS"].value_counts().plot(kind="bar")
champion.loc[10:15]
champion.iloc[5:10,0:7] 
champion.Win.nlargest(5)
champion.index
champion.loc[6]
champion_pieces = [champion[:3], champion[3:7], champion[7:5]]

champion_pieces
champion[champion['TPP'].notnull()].head()
champion[champion['TPP'].isnull()]
champion.ix[2, 'Team']
champion.Team.ix[2]
champion.describe()   
champion.info()   
champion.Team.str.len()
champion.groupby('Team').agg(['min', 'max'])
champion.groupby(['Team','Year']).count()
champion.groupby(['Year']).groups.keys()
len(champion.groupby(['Year']).groups[1980])