import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
matches = pd.read_csv('../input/WorldCupMatches.csv')
wcups = pd.read_csv('../input/WorldCups.csv')
players = pd.read_csv('../input/WorldCupPlayers.csv')
match = matches.dropna()
match.head()
match.loc[match['Attendance']== max(match['Attendance'])] 
match.loc[match['Attendance']== min(match['Attendance'])] 
match['City'].value_counts().head() 
match['Stadium'].value_counts().head()
A = match.groupby('Year')['Attendance'].sum().reset_index()
A['Year']=A['Year'].astype(int)

plt.figure(figsize=(12,8))
sns.barplot(y =A['Attendance'],x = A['Year'],edgecolor="k"*len(A))
plt.show()
Avg = match.groupby('Year')['Attendance'].mean().reset_index()
Avg['Year'] = Avg['Year'].astype(int)

sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
sns.pointplot('Year','Attendance',data=Avg,color='black')
plt.show()
Go = abs(match['Home Team Goals'] - match['Away Team Goals'])
match = match.assign(Goal_diff= Go)
match.loc[match['Goal_diff'] == max(match['Goal_diff'])]
match['Referee'].value_counts().head()
wcups
sns.set_style("whitegrid")
plt.figure(figsize=(13,8))
wcups["Year"] = wcups["Year"].astype(str)
plt.scatter(y =wcups['GoalsScored'],x = wcups['Year'],edgecolor="k"*len(A),alpha=.7,linewidth=2,s=500,color='black')
plt.xticks(wcups["Year"].unique())
plt.show()
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
sns.barplot(y='MatchesPlayed',x ='Year',data=wcups,color='white',edgecolor="k"*len(wcups))
