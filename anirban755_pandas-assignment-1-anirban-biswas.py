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
match=pd.read_csv('/kaggle/input/ipl/matches.csv')
match
# Q1. Find the number of matches played in Mumbai in the period of 2013 and 2017
mask = ((match['season']>=2013) & (match['season']<=2017))& (match['city']=='Mumbai')
match[mask]
len(match[mask])
# Q2. Find the number of matches where the margin of victory was greater than 30
mask= match['win_by_runs']>=30
match[mask]
len(match[mask])
# Q3. What percent times KKR decides to bat after winning the toss
mask1= (match['toss_winner']=='Kolkata Knight Riders')
totalwintos=len(match[mask1])
mask2= (match['toss_decision']=='bat')
totalbatting=len(match[mask1 & mask2])
percentageoftosswin= (totalbatting/totalwintos)*100
percentageoftosswin
# Q4 : Find the name of the player who won max number of man of the matches award in the period of 2010 and 2015 while plating in Mumbai [Easy]
mask1=(match['season']>=2010)&(match['season']<=2015)
mask2=match['player_of_match']
match[mask1&mask2]['player_of_match'].value_counts().head(1).index[0]
# Q5 : Find the team who has won most number of matches with victory margin > 50 [Easy]
mask = (match['win_by_runs']>=50)
match[mask]['winner'].value_counts().head(1).index[0]
# Q6 : Write a function which will take 2 inputs, team 1 and team 2 and their record against each other

# For example if team1->KKR and team2->CSK

# The output should be like KKR 2(matches won by KKR) and CSK 5(matches won by CSK)[Medium]
def teamvsteam(team1,team2):
    mask1=((match['team1']==team1)& (match['team2']==team2))
    mask2=((match['team1']==team2)& (match['team2']==team1))
    matches=(match[mask1 | mask2])
    totalmatches=(matches.shape[0])
    match_won_team1=(matches[matches['winner']==team1].shape[0])
    match_won_team2=(matches[matches['winner']==team2].shape[0])
    print("total no of matches is",totalmatches)
    print("won by {} is ".format(team1),match_won_team1)
    print("won by {} is ".format(team2),match_won_team2)
    print("no of matches drawn",(totalmatches-(match_won_team1+match_won_team2)))
    
    

teamvsteam('Mumbai Indians','Kolkata Knight Riders')
# Q7 : Write a function which will accept a team name as input and will return it's win percentage after winning the toss[Difficult] 
def winpercentage(team):
    mask1 = ((match['team1']==team) | (match['team2']==team))
    totalmatchplayed=match[mask1].shape[0]
    mask2 = ((match['winner']==team) & (match['toss_winner']==team))
    teamselect=match[mask2].shape[0]
    print("{} has won {} matches after the toss, & win percentage is".format(team,teamselect),((teamselect/totalmatchplayed)*100),'% out of total matches played by them.')
winpercentage('Mumbai Indians')


