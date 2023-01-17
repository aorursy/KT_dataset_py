import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px
ipl=pd.read_csv('../input/ipldata/matches.csv')
ipl
ipl.shape
ipl.columns
ipl.isna().any()
ipl.isnull().sum()
ipl.describe()
ipl['id'].count()
ipl['season'].unique()
fig_dims=(20,4)

plt.subplots(figsize=fig_dims)

sns.countplot(x=ipl['season'],data=ipl,palette='Set1')

plt.show()
#You may notice Banglore is at position 4 right now
sns.countplot(y="city", data=ipl, palette="Greens_d",

              order=ipl.city.value_counts().iloc[:5].index)



plt.show()
# Banglore is repeated as Benguluru which is due to improper data cleaning wich can be fixed 
ipl['city'].unique()
replace={'Bengaluru':'Bangalore'}

ipl.city.replace(replace, inplace=True)

ipl['city'].unique()
#Now banglore has jumped to 2nd position based on on number of games hosted



#This show cases the importance of data cleaning



sns.countplot(y="city", data=ipl, palette="Greens_d",

              order=ipl.city.value_counts().iloc[:5].index)

plt.show()
sns.countplot(x="player_of_match", data=ipl, palette="rocket",

              order=ipl.player_of_match.value_counts().iloc[:3].index)

plt.show()
#Rising super giants occurs as Rising super giant due to data entry issue

ipl['winner'].value_counts()
#Replacing duplicate team name 'Rising Pune Supergiant' as 'Rising Pune Supergiants'

replace={'Rising Pune Supergiant':'Rising Pune Supergiants'}

ipl.winner.replace(replace, inplace=True)

ipl['winner'].unique()
ar=ipl['winner'].value_counts().keys()

list1 = ar.tolist()
ar2=ipl['winner'].value_counts()

list2 = ar2.tolist()
px.pie(ipl,values=list2,names=list1,title='Percentage of wins')

ipl1=ipl[ipl['toss_winner']==ipl['winner']]

sns.countplot(y="winner", data=ipl1, palette="Reds_d",

              order=ipl1.winner.value_counts().index)

plt.show()
fig=plt.gcf()

fig.set_size_inches(15,10)

sns.swarmplot(ipl['season'],ipl[ipl['win_by_runs']!=0]['win_by_runs'],s=10)

plt.title('Season wise match summary of matches won by runs')

plt.xlabel('Season')

plt.ylabel('Runs')

plt.show()
plt.figure(figsize=(10,10))



ipl1=ipl[ipl['toss_winner']==ipl['winner']]



sns.countplot("winner", data = ipl1, hue = 'toss_decision',order = ipl1['toss_winner'].value_counts().index,palette='Set1')

plt.title("Wins on choosing to field first vs choosing to bat first for each franchise b/w 2008-2019")

plt.xticks(rotation=45, ha = 'right')

plt.ylabel('Number of Matches')

plt.show()