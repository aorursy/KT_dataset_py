# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
matches=pd.read_csv('../input/matches.csv')   
delivery=pd.read_csv('../input/deliveries.csv')
matches.head(10)

delivery.head(10)

matches.columns
import matplotlib.pyplot as mlt
toss_winners = matches['toss_winner']
winner = matches['winner']

cnt = 0
for a in range(len(toss_winners)):
    if toss_winners[a] != winner[a] :
        cnt += 1       
print(str(cnt)+ " out of " + str(len(toss_winners))+ " times, the team who won the toss had won the match")        
print("Graph showing the number of times a team has won the toss")
mlt.subplots(figsize=(10,6))
ax=matches['toss_winner'].value_counts().plot.bar(width=0.8)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
mlt.show()


print("Graph showing the number of times a team has won the match")

ax=matches['winner'].value_counts().plot.bar(width=0.8)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
mlt.show()

mlt.subplots(figsize=(10,6))
matches['toss_winner'].value_counts().plot.bar(width=0.10)
matches['winner'].value_counts().plot.line()
mlt.xticks(rotation=90)
mlt.show()


print("The above graph leads to the conclusion that CSK,DD,SRH have emerged to the winners whenever they have won the toss")
newpd = matches.loc[matches['toss_winner'] == matches['winner']]
newpd = newpd.groupby(['toss_winner','toss_decision'])
newpd.size()

newpd['toss_decision'].value_counts().plot.bar(width=0.10)
mlt.xticks(rotation=90)
mlt.show()

print("Conclusions from the bove graph are..")
print("CSK has higher chances of winning if it goes for batting first")
print("DD, Kings XI Punjab, Kolkata Night Riders, Rajasthan Royals, RCB  has higher chances of winning if it goes for fielding first and are good chasers")


newpd = delivery.groupby(['batsman', 'non_striker']).agg({'total_runs':'sum' }).sort_values(by = 'total_runs',ascending=False)
newpd = newpd.head(10)
newpd
newpd.plot.bar(width=0.10)
mlt.xticks(rotation=90)
mlt.show()
print("This shows best patnership is Kohli's patnership with gayle and AB de villiers")
newpd = delivery.groupby(['bowler']).agg({'noball_runs':'sum' }).sort_values(by = 'noball_runs',ascending=False)
newpd.head(10)


newpd = delivery.groupby(['bowler','player_dismissed'])
df = newpd.size()
df.sort_values(ascending=False).head(10)
delivery1 = delivery.loc[delivery['dismissal_kind'] == 'bowled']
newpd = delivery1.groupby(['bowler','dismissal_kind'])
df = newpd.size()
df.sort_values(ascending=False).head(10)
delivery2 = delivery.loc[delivery['dismissal_kind'] == 'caught']
newpd = delivery2.groupby(['bowler','dismissal_kind'])
df = newpd.size()
df.sort_values(ascending=False).head(10)
print("Analysis of player of the match")

newpd = matches['player_of_match'].value_counts()
newpd.head(5)
delivery.fillna(0,inplace=True) 
delivery_1 = delivery.loc[(delivery["over"] >= 15) & delivery["player_dismissed"] != 0]
print(delivery_1.groupby(["bowler"]).size().sort_values(ascending=False).head(10))
delivery_1.groupby(["bowler"]).size().sort_values(ascending=False).head(10).plot.bar(width=0.8)
delivery.fillna(0,inplace=True) 
delivery_2 = delivery.loc[(delivery["over"] >= 15)].groupby(['batsman']).agg({'total_runs':'sum'})
print(delivery_2.sort_values(by='total_runs',ascending=False).head(10))
delivery_2.sort_values(by='total_runs',ascending=False).head(10).plot.bar(width=0.8)

matches.groupby(['winner','season']).size().head(10)

matches_2 = matches.groupby(['season','winner']).size().to_frame().unstack(level=-1)
matches_2.plot(figsize=(15,10))
print("Graph showing the performance the teams over the years in terms of the number of matches won.")
new_df = matches.loc[matches["team1"] != matches["winner"]].groupby(["team1","team2"]).size().sort_values(ascending=False).head(5)
new_df
