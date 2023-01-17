# Importing the libraries



import pandas as pd

import numpy as np

import seaborn as sns
# Retrieving the first dataset



runs = pd.read_csv(r'../input/indian-premier-league-csv-dataset/Ball_by_Ball.csv')

runs.head()
# Retrieving the second dataset



Players = pd.read_csv(r'../input/indian-premier-league-csv-dataset/Player.csv')

Players = Players.drop(["Is_Umpire", "Unnamed: 7"], axis = 1)

# Players = Players[["Player_Id", "Player_Name"]]

Players.head()
# Keeping only the relevant data for analysis



runs = runs[["Match_Id", "Over_Id", "Striker_Id", "Bowler_Id", "Batsman_Scored", "Extra_Runs"]]

runs.head()
# Checking for null values



runs.isnull().sum()
# Since isnull() gives no null values, we check for data types.



runs.dtypes
# Replacing null values with 0 runs.



runs["Extra_Runs"] = pd.to_numeric(runs["Extra_Runs"], errors="coerce")

runs["Extra_Runs"] = runs["Extra_Runs"].fillna(0)



runs["Batsman_Scored"] = pd.to_numeric(runs["Batsman_Scored"], errors="coerce")

runs["Batsman_Scored"] = runs["Batsman_Scored"].fillna(0)

runs.head(14)
A = []

B = []



for i in runs["Striker_Id"].unique():

    a = runs["Batsman_Scored"][runs["Striker_Id"] == i].sum()

    b = runs["Match_Id"][runs["Striker_Id"] == i].count()

    A.append(a)

    B.append(b)



Batsmen = pd.DataFrame({"Player_names": runs["Striker_Id"].unique()})



Batsmen["Runs"] = A

Batsmen["Balls_played"] = B



# Calculating strike rate using the formula (strike_rate = runs per 100 balls faced)



Batsmen["Strike_Rate"] = (Batsmen["Runs"] * 100 / Batsmen["Balls_played"]).round(2)

# Checking the distribution of batsmen



Batsmen["Strike_Rate"].hist(color="pink")
Batsmen = Batsmen[Batsmen["Strike_Rate"] > 60]

Batsmen = Batsmen[Batsmen["Strike_Rate"] < 150]
Batsmen = Batsmen[(Batsmen["Runs"]*6/Batsmen["Balls_played"]) > 6]
Batsmen.head()
Batsmen["Strike_Rate"].hist(color="pink")
import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style("whitegrid")

fig = plt.figure(figsize=(12,6))

plt.scatter(Batsmen["Strike_Rate"], Batsmen["Runs"], color="red")

plt.xlabel("Strike Rate of the Batsmen", fontsize=14)

plt.ylabel("Runs scored by the Batsmen", fontsize=14)

plt.title("IPL Batsmen", fontsize=16)

plt.show()
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=5)

kmeans.fit(Batsmen[["Strike_Rate", "Runs"]])

Batsmen["cluster"] = kmeans.labels_



fig = plt.figure(figsize=(12,6))



colors = ["blue", "sienna", "limegreen", "red", "black"]



for r in range(1,6):

    clustered_Batsmen = Batsmen[Batsmen["cluster"] == r]

    plt.scatter(clustered_Batsmen["Strike_Rate"], clustered_Batsmen["Runs"], color=colors[r-1])

    

plt.title("IPL Batsmen", fontsize=16)

plt.xlabel("Strike Rate of the Batsmen", fontsize=14)

plt.ylabel("Runs scored by the Batsmen", fontsize=14)

    

plt.show()
Batsmen = Batsmen.rename(columns={"Player_names": "Player_Id"})

Batsmen = Batsmen.sort_values(by ='cluster' )

Batsmen.head()
Players_batsmen = Players[["Player_Id", "Player_Name","Batting_Hand","DOB","Country"]]
Batsmen = Batsmen[Batsmen.columns].merge(Players_batsmen, "left")

Batsmen
def plot_bar(df, feat_x, feat_y, normalize=True):

    """ Plot with vertical bars of the requested dataframe and features"""

    

    ct = pd.crosstab(df[feat_x], df[feat_y])

    if normalize == True:

        ct = ct.div(ct.sum(axis=1), axis=0)

    return ct.plot(kind='bar', stacked=True)
plot_bar(Batsmen, 'Batting_Hand', 'cluster')

plt.show()
Batsmen.to_csv('/kaggle/working/Batsmen.csv') 
C = []

E = []



for j in runs["Bowler_Id"].unique():

    c = runs["Batsman_Scored"][runs["Bowler_Id"] == j].sum() + runs["Extra_Runs"][runs["Bowler_Id"] == j].sum()

    

    e = runs["Over_Id"][runs["Bowler_Id"] == j].count()/6

    

    C.append(c)

    E.append(e)

    

Bowlers = pd.DataFrame({"Bowler_names": runs["Bowler_Id"].unique()})



Bowlers["Runs"] = C

Bowlers["Over_count"] = E



# Calculating economy rate using the formula (total runs conceded/number of overs bowled)



Bowlers["Econ_Rate"] = (Bowlers["Runs"] / Bowlers["Over_count"]).round(2)



Bowlers.head()
# Checking the distribution of batsmen 



Bowlers["Econ_Rate"].hist(color="pink")
Bowlers = Bowlers[(Bowlers["Econ_Rate"] > 4) & (Bowlers["Econ_Rate"] < 10.5)]

Bowlers["Econ_Rate"].hist(color="pink")
sns.set_style("whitegrid")



fig = plt.figure(figsize=(12,6))

plt.scatter(Bowlers["Econ_Rate"], Bowlers["Runs"], color="teal")

plt.xlabel("Economy Rate of the Bowler", fontsize=14)

plt.ylabel("Number of Overs bowled by the Bowler", fontsize=14)

plt.title("IPL Bowlers", fontsize=16)

plt.show()
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=5)

kmeans.fit(Bowlers[["Econ_Rate", "Over_count"]])

Bowlers["cluster"] = kmeans.labels_



fig = plt.figure(figsize=(12,6))



colors = ["blue", "lightgreen", "black", "red"]



for r in range(1,5):

    clustered_Bowlers = Bowlers[Bowlers["cluster"] == r]

    plt.scatter(clustered_Bowlers["Econ_Rate"], clustered_Bowlers["Over_count"], color=colors[r-1])

    

plt.title("IPL Bowlers", fontsize=16)

plt.xlabel("Economy Rate of the Bowler", fontsize=14)

plt.ylabel("Number of Overs bowled by the Bowler", fontsize=14)

    

plt.show()
Bowlers = Bowlers.rename(columns={"Bowler_names": "Player_Id"})

Bowlers = Bowlers.sort_values(by ='cluster' )

Bowlers.head()
Players_bowlers = Players[["Player_Id", "Player_Name","Bowling_Skill","DOB","Country"]]
Bowlers = Bowlers[Bowlers.columns].merge(Players_bowlers, "left")

Bowlers
plot_bar(Bowlers, 'Bowling_Skill', 'cluster')

plt.show()
Bowlers.to_csv('/kaggle/working/Bowlers.csv') 
final=pd.concat([Batsmen,Bowlers],sort="True").reset_index(drop=True)

final
final.to_csv('/kaggle/working/final.csv')
batsmen_team=Batsmen[['Player_Name','Strike_Rate','DOB','Country']]

batsmen_team=batsmen_team.head(5)

batsmen_team
bowlers_team=Bowlers[['Player_Name','Econ_Rate','DOB','Country']]

bowlers_team=bowlers_team.head(5)

bowlers_team
d1 = pd.merge(Batsmen,Bowlers, how='inner', on=['Player_Id'])

d2 = pd.merge(d1,Players, on='Player_Id')

all_rounder_team=d2[['Player_Name_x','Strike_Rate','Econ_Rate','DOB','Country']]

all_rounder_team.rename(columns = {'Player_Name_x':'Player_Name'}, inplace = True) 

all_rounder_team=all_rounder_team.head(10)

all_rounder_team
df=pd.concat([batsmen_team,bowlers_team,all_rounder_team],sort=True).drop_duplicates(subset='Player_Name', keep="first").reset_index(drop=True)

df=df[['Player_Name','Strike_Rate','Econ_Rate','DOB','Country']]

print("-"*30,"Well rounded team","-"*30,"\n")

df