import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
ipl=pd.read_csv('../input/ipl-cricket-dataset/matches.csv') #for reading csv 
ipl.head() #first five records ofthe data frame 
ipl.shape #glance at the shape method i.e it give the no of records and no of column
ipl['player_of_match'].value_counts() # it will give how much time a person gets a player of the match awards.
#here is dataframe is ipl and name of the colum is player_of_match and func value.counts() 

ipl['player_of_match'].value_counts()[0:10] #for top 10 records only
plt.figure(figsize=(7,7))
plt.bar(list(ipl['player_of_match'].value_counts()[0:5].keys()),list(ipl['player_of_match'].value_counts()[0:5]),color=["blue","green","pink","orange","red"])
plt.show() # from first line here we are setting the figure size i.e the bar graph thicknes.
#the second line gives the bar in x axis gives the player name and in y axis gives the valuecounts of the matches. 
ipl['result'].value_counts()
ipl['toss_winner'].value_counts()
batting_first=ipl[ipl['win_by_runs']!=0]
batting_first.head() #semidata frame which i have created from original data frame
plt.figure(figsize=(7,7))
plt.hist(batting_first['win_by_runs'])
plt.show()
plt.figure(figsize=(7,7))
plt.bar(list(batting_first['winner'].value_counts()[0:3].keys()),list(batting_first['winner'].value_counts()[0:3]),color=["blue","yellow","red"])
plt.show()
plt.figure(figsize=(7,7))
plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()
batting_second=ipl[ipl['win_by_wickets']!=0] #
batting_second.head()
plt.figure(figsize=(7,7))
plt.hist(batting_second['win_by_wickets'],bins=30)# bins  will give much more clear images 
plt.show()
plt.figure(figsize=(7,7))
plt.bar(list(batting_second['winner'].value_counts()[0:3].keys()),list(batting_second['winner'].value_counts()[0:3]),color=["blue","yellow","red"])
plt.show()
plt.figure(figsize=(7,7))
plt.pie(list(batting_second['winner'].value_counts()),labels=list(batting_second['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()
ipl['season'].value_counts()
ipl['city'].value_counts()
import numpy as np
np.sum(ipl['toss_winner']==ipl['winner']) #it will which has won the toss also win the match
325/636
deliveries=pd.read_csv('../input/ipl-cricket-dataset/deliveries.csv')
deliveries.head()
deliveries['match_id'].unique()
match_1=deliveries[deliveries['match_id']==1]
match_1.head()
match_1.shape
srh=match_1[match_1['inning']==1]
srh['batsman_runs'].value_counts()

srh["dismissal_kind"].value_counts()
rcb=match_1[match_1['inning']==2]
rcb['batsman_runs'].value_counts()
rcb['dismissal_kind'].value_counts()
bowling=pd.read_csv('../input/ipl-cricket-dataset/Bowlers.csv')
bowling.head()
bowling.shape
bowling.tail()
bowling.head(3)
bowling.tail(3)
bowling[2:8] #we can get the 2nd row to 8th row excluding 8th row 
bowling[0::2] # it will give the alternate rows
bowling[5:0:-1] #display the rows in reverse order,we can use negative step size in slicing
bowling.columns #to retrieveing columns name
bowling.Econ
bowling['Econ']
bowling['wickets'].max()
bowling['wickets'].min()
bowling.describe() #it displays very important information like number of values,average,min,max 25%,50%,75% of the total value 
bowling[bowling.runs>50]
#retrieve the row where the row  is maximum
bowling[bowling.runs==bowling.runs.max()]
bowling[['over','wickets']][bowling.runs>50] # display only the id numbers and names where the salary is greater
bowling.index
df1=bowling.set_index("over")
df1
bowling.set_index('over',inplace=True)
bowling
bowling.loc[4]
#resetting the index
bowling.reset_index(inplace=True)
bowling
#sorting the data
df=pd.read_csv('../input/ipl-cricket-dataset/matches1234.csv')
df
df1=df.sort_values('win_by_runs')
df1
df1.tail()
df1=df.sort_values('win_by_runs',ascending=False)
df1
df1=df.sort_values(by=['win_by_runs','win_by_wickets'],ascending=[False,True]) #i.e when two team have same win_by_runs then their wickets will be sorted into ascending order
df1
plt.figure(figsize=(7,7))
x=df['team1']
y=df['team2']
plt.bar(x,y,label='win_by_runs',color='black')
plt.xlabel("Team1")
plt.ylabel("Team2")
plt.title('GRAPH FOR WINNING TEAM')
plt.legend()
plt.show()
plt.figure(figsize=(7,7))
plt.hist(df['win_by_runs'])
plt.show()
plt.figure(figsize=(7,7))
plt.pie(list(df['win_by_runs'].value_counts()),labels=list(df['win_by_runs'].value_counts().keys()),autopct='%0.1f%%')
plt.show()
