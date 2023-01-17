import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ipl_data=pd.read_csv("../input/ipl/matches.csv")
ipl_data.head(10)
ipl_data.info()
#No of Match played in top 10 city
played_city=ipl_data['city'].value_counts().sort_values(ascending=False)
played_city=played_city.head(10)
sns.set_style('darkgrid')
plt.figure(figsize=(20,6))
noofmatch=sns.barplot(played_city.index,played_city.values)
plt.title("No of Match played in city",fontsize=15)
plt.ylabel("No of Match",fontsize=12)
plt.xlabel("Team",fontsize=15)
noofmatch.set_xticklabels(rotation=50,labels=played_city.index,fontsize=13)
noofmatch.set_yticklabels(fontsize=13,labels=played_city.values)
plt.show()


#Most Played Venue
sns.set_style("darkgrid")
ls=ipl_data['venue'].value_counts().sort_values(ascending=False)
ls=ls[:7]
plt.figure(figsize=(20,6))
Most_played=sns.barplot(ls.index,ls.values,alpha=0.8)
plt.title("Most Played Venue")
plt.ylabel("count",fontsize=12)
plt.xlabel("No of matches",fontsize=12)
Most_played.set_xticklabels(rotation=30,labels=ls.index,fontsize=10)
plt.show()
#Man of Match Player
man_of_match=ipl_data['player_of_match'].value_counts().sort_values(ascending=False)
man_of_match=man_of_match[:9]
sns.set_style("darkgrid")
plt.figure(figsize=(20,6))
man_of_matches=sns.barplot(man_of_match.index,man_of_match.values,alpha=0.8,palette='winter')
plt.title("Most of Time Man of Match player")
plt.ylabel("count",fontsize=12)
plt.xlabel("Player",fontsize=12)
man_of_matches.set_xticklabels(rotation=30,labels=man_of_match.index,fontsize=15)
plt.show()
#played Match in every season
plt.figure(figsize=(20,6))
season=sns.countplot(x='season',data=ipl_data)
plt.xlabel('season',fontsize=20)
plt.show()
#Favourite Umpire
ump=pd.concat([ipl_data['umpire1'],ipl_data['umpire2']]).value_counts().sort_values(ascending=False)
ump=ump[:7]
plt.figure(figsize=(20,5))
Most_umpired=sns.barplot(x=ump.index,y=ump.values,alpha=0.9)
plt.title("Favourite Umpire")
plt.ylabel("Count",fontsize=15)
plt.xlabel("Umpire Name",fontsize=15)
Most_umpired.set_xticklabels(rotation=50,labels=ump.index,fontsize=20)
plt.show()
#No of Match win Every team
noofwin=ipl_data['winner'].value_counts().sort_values(ascending=False)
sns.set_style("darkgrid")
plt.figure(figsize=(20,6))
Mostwin=sns.barplot(x=noofwin.index,y=noofwin.values,alpha=0.9)
plt.title("No Of Match Wins")
plt.xlabel("Team Name",fontsize=14)
plt.ylabel("count",fontsize=14)
Mostwin.set_xticklabels(rotation=50,labels=noofwin.index,fontsize=10)
plt.show()
#Match played in every season using hist 
plt.hist(ipl_data['season'],bins=20)
plt.show()
#toss winner
ss=ipl_data['toss_winner'].value_counts()
plt.figure(figsize=(20,6))
tosswinner=sns.barplot(ss.index,ss.values)
plt.xlabel("Team",fontsize=12)
plt.ylabel("count",fontsize=12)
tosswinner.set_xticklabels(rotation=50,labels=ss.index)
plt.show()

#winning the toss and take a decision
ipl_data1=ipl_data['toss_decision'].value_counts()
plt.pie(ipl_data1,labels=ipl_data1.index)
plt.legend()
plt.show()
#winning the toss and take decision every season
plt.figure(figsize=(20,6))
toss=sns.countplot('season',hue='toss_decision',data=ipl_data)