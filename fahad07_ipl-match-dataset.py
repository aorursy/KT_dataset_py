import pandas as pd
df=pd.read_csv('../input/ipl-match-database/matches.csv')
df
df['team1'].unique()
# select distinct in sql 
df['team1'].value_counts().plot(kind ='bar')
df['player_of_match'].head(10).value_counts().plot(kind='pie',figsize=[8,8])
dfm1=df[(df['team1']=='Mumbai Indians')&(df['team2']=='Chennai Super Kings')]
dfc1=df[(df['team2']=='Mumbai Indians')&(df['team1']=='Chennai Super Kings')]
len(dfm1)
len(dfc1)
dfm1['winner'].value_counts()
dfc1['winner'].value_counts()
df[(df['city']=='Mumbai')|(df['city']=='Delhi')]
dfboth=df[((df['team1']=='Mumbai Indians')&(df['team2']=='Chennai Super Kings'))|((df['team1']=='Chennai Super Kings')&(df['team2']=='Mumbai Indians'))]
len(dfboth)
dfboth['winner'].value_counts()
dfboth['winner'].value_counts().plot(kind='pie',figsize=[8,8])
t1 = input('Enter Team One:')
t2 = input('Enter Team Two:')
tg = input("Enter Type of Graph")
dfboth=df[((df['team1']==t1)&(df['team2']==t2))|((df['team1']==t2)&(df['team2']==t1))]
dfboth['winner'].value_counts().plot(kind=tg,figsize=[8,8])
