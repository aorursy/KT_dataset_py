# **** SPORTS ANALYTICS ****
# this is called sports analytics
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
import pandas as pd
#how to import csv file
match = pd.read_csv('/kaggle/input/ipl/matches.csv')
match
#pandas
#--->dataframe
#--->series
#what are the first things to do?
match.shape

#how to take data preview
match.head(3)
match.tail() #default is 5
match.sample() #randomly picks row
#overall info of the dataframe
match.info()
match.describe()
match['city']
match['city'].shape

match['city'].sample
#how to fetch multiple columns
match[['team1','team2','winner']]
#how to fetch single row from data frame
#match.iloc[5]
#match.iloc[5].values
match.iloc[5].shape
#how to fetch multiple row from a data frame
match.iloc[3:7]
match.iloc[3:7]['city']
match.head()
#frequency of all 
match['winner'].value_counts()
#max winner
match['winner'].value_counts().head(1).index[0]
#max player of match winner
match['player_of_match'].value_counts().head(1).index[0]
#how to filter a data frame based on a given criteria
mask=match['city']=="Kolkata"
match[mask]['player_of_match'].value_counts().head(1).index[0]
mask1=match['city']=='Hyderabad'
mask2=match['season']>=2015
match[mask1 & mask2]

#how to filter a dataframe on the basis of multiple conditions
match[mask1 & mask2]['player_of_match'].value_counts()
#plot graph
#sort vaues
#drop_duplicates
#problems
import matplotlib.pyplot as plt
#plot graph
match['winner'].value_counts().head(5).plot(kind='barh') #top 5 teams
#if you have series you can use plot. here bar plot
#bar = normal bar chart
#barh =horizontal barchart
match['toss_decision'].value_counts()
match['toss_decision'].value_counts().plot(kind='pie')
match.drop_duplicates(subset=['season'])
#match['toss_decision'].value_counts().sum()
(match['toss_decision'].value_counts()/match['toss_decision'].value_counts().sum())*100
#no of matches played in mumbai
match[match['city']=="Mumbai"].shape[0]
#shape gives shape of the data and 0th index is row and 1st index is coulumn. we need rows hence did shape[0]
new=match[match['city']=="Mumbai"]
new['winner'].value_counts().index[1]
#valuecounts counts the value ie each team won how many match
#index returns only the team names
#we need second highest winner, hence index 1
new1=match[match['city']=="Kolkata"]
new1['player_of_match'].value_counts().index[0]
new1=match[(match['city']=="Kolkata") & (match['season']==2015)]
new1['player_of_match'].value_counts().index[0]

    
def teamVteam(team1,team2):
    mask1=(match['team1']==team1) & (match['team2']==team2)
    mask2=(match['team1']==team2) & (match['team2']==team1)
    new=match[mask1 | mask2]  
    total_matches=new.shape[0]
    team1winner=new[new['winner']==team1].shape[0]
    team2winner=new[new['winner']==team2].shape[0]
    print("No of matches: ",total_matches)
    print("Matches won by {} : ".format(team1),team1winner)
    print("Matches won by {} : ".format(team2),team2winner)
    print("Matches drawn : ",total_matches-(team1winner+team2winner))
    

    
    
teamVteam('Kings XI Punjab','Royal Challengers Bangalore')
mask1=(match['city']=='Mumbai') & (match['season']==2013) 
mask2=(match['city']=='Mumbai') & (match['season']==2017)
new=match[mask1 | mask2]  
#new.shape[0]
mask1=match[match['win_by_runs']>30]
mask1.shape[0]
mask1=(match['team1']=='Kolkata Knight Riders')
mask2=(match['team2']=='Kolkata Knight Riders')
new1=match[mask1 | mask2]
total_match=new1.shape[0]

mask3=(match['toss_winner']=='Kolkata Knight Riders')
mask4=(match['toss_decision']=='bat')
new2=match[mask3 & mask4]
total_bat=new2.shape[0]

(total_bat/total_match)*100


mask1=(match['city']=='Mumbai')
mask2=(match['season']>=2010) & (match['season']<=2015)
new=match[mask1 | mask2]
new['player_of_match'].value_counts().index[0]
new=match[match['win_by_runs']>50]
new['winner'].value_counts().index[0]

#function which will accept a team name as input and will return it's 
#win percentage after winning the toss[Difficult] 

def teamwinpercentage(team):
    mask1=(match['toss_winner']==team)
    mask2=(match['winner']==team)
    new=match[mask1 & mask2] 
    winner=new.shape[0] #winner after winning toss
    tosswinner=(match['toss_winner']==team).shape[0]
    winpercentage=(winner/tosswinner)*100
    return winpercentage
    
teamwinpercentage('Chennai Super Kings')


mask1=(match['toss_winner']=='Kolkata Knight Riders')
mask1
myseries=match['winner'].value_counts()
myseries
myseries['Pune Warriors']
(match['team1']).value_counts() + (match['team2']).value_counts()
(((match['team1']).value_counts() + (match['team2']).value_counts()).head()).sort_values()
(((match['team1']).value_counts() + (match['team2']).value_counts()).head()).sort_values(ascending=False)
match.sort_values('city',ascending=False)
#match.sort_values('city',ascending=False,inplace=True) inplace with make the permanent change in data
match.sort_values(['city','date'],ascending=[True,False])
#sorting acc to city AND date acc to asc and desc respectively
match.drop_duplicates(subset=['city'])
#match.drop_duplicates(subset=['city']).shape[0]
#ipl aajtak 31 unique cities mei khela gaya
match.drop_duplicates(subset=['city'],keep='last')
#keep last keeps the last occurence of data, by default its first
match.drop_duplicates(subset=['city','season'],keep='last')
delivery=pd.read_csv('/kaggle/input/ipl/deliveries.csv')
delivery.head(7)
#total runs scored by virat
mask=delivery['batsman']=='MS Dhoni'
delivery[mask]['batsman_runs'].sum()
#IPL top 5 batsman(most no of runs scored)
delivery['batsman'].unique()
batsman=delivery['batsman'].unique()
#for i in batsman:
 #   mask=delivery['batsman']==i
  #  delivery[mask]['batsman_runs'].sum()
#IPL top 5 batsman(most no of runs scored)
delivery.groupby('batsman').sum()['batsman_runs'].sort_values(ascending=False)
#who hit most six in ipl
six=delivery[delivery['batsman_runs']==6]
six.shape #total no of sixes in ipl
six.groupby('batsman').count()['batsman_runs'].sort_values(ascending=False).head()

#who hit most fours in ipl
four=delivery[delivery['batsman_runs']==4]
four.shape #total no of sixes in ipl
four.groupby('batsman').count()['batsman_runs'].sort_values(ascending=False).head()

#max runs scored by batsman against which team
def batsmanVteam(batsman):
    return delivery[delivery['batsman']==batsman].groupby('bowling_team').sum()['batsman_runs'].sort_values(ascending=False).head(1).index[0]
batsmanVteam('MS Dhoni')
#max runs scored by batsman against which team
def batsmanVbowler(batsman):
    return delivery[delivery['batsman']==batsman].groupby('bowler').sum()['batsman_runs'].sort_values(ascending=False).head(1).index[0]

batsmanVbowler('MS Dhoni')
six=delivery[delivery['batsman_runs']==6]
pt=six.pivot_table(index='over',columns="batting_team",values='batsman_runs',aggfunc='count')
import seaborn as sns

sns.heatmap(pt,cmap='summer')
#winter,autumn different themes
#how to merge 2 or more dataframes

match=pd.read_csv("/kaggle/input/ipl/matches.csv")
merged=delivery.merge(match,left_on='match_id',right_on='id')
#delivery is merged with match not vice versa
merged

def plotcarriercurve(batsman):
    merged[merged['batsman']==batsman].groupby('season').sum()['batsman_runs'].plot(kind='line')
plotcarriercurve("DA Warner")