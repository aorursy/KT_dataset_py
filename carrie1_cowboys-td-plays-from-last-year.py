import pandas as pd

df = pd.read_csv('../input/NFL by Play 2009-2016 (v2).csv',low_memory=False)

df.info() 
#find Dallas games from last year

cowboys = df[((df["HomeTeam"] == 'DAL') | (df["AwayTeam"] == 'DAL')) 

             & (df["Season"] == 2016) & (df['Touchdown'] == 1)]

grouped = cowboys.groupby(by='Date') 

len(grouped) #confirm by counting game dates
#find plays where Dallas is on offense

offense = cowboys[(cowboys["DefensiveTeam"] != 'DAL')]



#find the top 25 longest plays by yards

longest_tds = offense.sort_values(by='Yards.Gained',ascending=False)[:25]



#make a variable that combines receivers and rushers

longest_tds['scorer'] = longest_tds['Rusher']

longest_tds['scorer'].fillna(longest_tds['Receiver'], inplace=True)



#check out the plays

longest_tds[['PlayType','down','Yards.Gained','Date','qtr','desc','scorer','Rusher','Receiver']]
#let's see who has the most TDs out of the 25 longest

import seaborn as sns

%matplotlib inline

sns.countplot(x="scorer", data=longest_tds)
#play type breakdown of top 25 plays

sns.countplot(x="PlayType", data=longest_tds)
#looking at passes only, who had the most TDs in all Dallas games

passes =  offense[(offense["PlayType"] == 'Pass')]

sns.countplot(x="Receiver", data=passes)
#looking at run plays only, who had the most TDs in all Dallas games

runs =  offense[(offense["PlayType"] == 'Run')]

sns.countplot(x="Rusher", data=runs)
#find all elliot TDs

elliott = offense[((offense["Rusher"] == 'E.Elliott') | (offense["Receiver"] == 'E.Elliott'))]



#find his average yards gained on a touchdown 

import numpy as np

np.mean(elliott['Yards.Gained'])
#compare with team 

np.mean(offense['Yards.Gained'])