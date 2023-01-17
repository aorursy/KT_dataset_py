import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df=pd.read_csv("/kaggle/input/ipl-data-set/matches.csv")
df.head()#exploring our data set
df.shape#checking no of rows and columns of our dataset
plt.subplots(figsize=(12,6))

df['umpire1'].value_counts().plot.bar(width=0.9,color="red",alpha=0.75)

plt.xlabel("Name of the umpires")

plt.ylabel("No.of matches as umpire ")

#"most capped umpire"
plt.subplots(figsize=(12,6))

df['winner'].value_counts().plot.bar(width=0.9,color="orange",alpha=0.75)

plt.xlabel("Name of the teams")

plt.ylabel("No.of wins ")

#team with most wins
plt.subplots(figsize=(12,6))

df['venue'].value_counts().plot.bar(width=0.9,color="pink",alpha=0.75)

#stadiums with maximum matches
plt.subplots(figsize=(12,6))

x=df['player_of_match'].value_counts()

y=x.head(40)

y.plot.bar(width=0.9,color="cyan",alpha=0.75)

#players with most player_of_the_match(top-40)
df1=pd.read_csv("/kaggle/input/ipl-data-set/teamwise_home_and_away.csv")

df1
df1.drop([0,11,12,13],axis="rows",inplace=True)#dropping all the incositent teams
df1
df1=df1.append({"team":"Delhi_Capitals","home_wins":"28","away_wins":"49","home_matches":"78","away_matches":"99",

                "home_win_percentage":"42.36111","away_win_percentage":"58.50000"},ignore_index=True)



df1=df1.append({"team":"Sunrisers_Hyderabad","home_wins":"48","away_wins":"39","home_matches":"106","away_matches":"77",

                "home_win_percentage":"44.73","away_win_percentage":"48.29"},ignore_index=True)



df1.drop([2,3,5,9],axis="rows",inplace=True)
df1 



#we have merged teams from delhi and hyderabad at their current franchise name &

    #  calculated the average of win and loss percentage
plt.subplots(figsize=(18,6))

plt.scatter(df1["team"],df1["home_win_percentage"],linewidths=5,color="red")

plt.xlabel("Teams")

plt.ylabel("WinPercentage")#scatter plot showing home win percentage
plt.subplots(figsize=(18,6))

plt.scatter(df1["team"],df1["away_win_percentage"],linewidths=5,color="purple")

plt.xlabel("Teams")

plt.ylabel("WinPercentage")#scatter plot showing away win percentage
df3=pd.read_csv("/kaggle/input/ipl-data-set/most_runs_average_strikerate.csv")

df3.head()
filt=(df3["total_runs"]>=4000)

df4=df3[filt]#gathering the stats of those players who scored more than 4000 runs in IPL
df4
plt.subplots(figsize=(19,6))

plt.bar(df4["batsman"],df4["total_runs"],color="yellow",alpha=0.75)