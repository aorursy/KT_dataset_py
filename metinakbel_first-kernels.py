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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("../input/PUBG_Player_Statistics.csv")
df.corr()
df.head()
df.info()
df.describe()
df.columns
plt.plot(df.solo_Top10s,df.solo_Wins,".",label="solo")
plt.legend()
plt.title("Plot Solo Top10s-Wins")
plt.xlabel("Top10s")
plt.ylabel("Wins")
plt.show()
plt.plot(df.duo_Top10s,df.duo_Wins,".",color="red",label="Duo")
plt.legend()
plt.title("Plot Duo Wins-Top10Win")
plt.xlabel("Top10s")
plt.ylabel("Wins")
plt.show()
plt.plot(df.squad_Top10s,df.squad_Wins,".",color="orange",label="Squad")
plt.legend()
plt.title("Plot Squad Top10s-Wins")
plt.xlabel("Top10s")
plt.ylabel("Wins")
plt.show()
plt.plot(df.solo_Wins,df.solo_VehicleDestroys,".",color="blue",label="Solo")
plt.legend()
plt.title("Plot Solo Wins-VehicleDest")
plt.xlabel("Wins")
plt.ylabel("VehicleDest")
plt.show()
plt.plot(df.duo_Wins,df.duo_VehicleDestroys,".",color="red",label="Duo")
plt.legend()
plt.title("Plot Duo Wins-VehicleDest")
plt.xlabel("Wins")
plt.ylabel("VehicleDest")
plt.show()
plt.plot(df.squad_Wins,df.squad_VehicleDestroys,".",color="orange",label="Squad")
plt.legend()
plt.title("Plot Squad Wins-VehicleDest")
plt.xlabel("Wins")
plt.ylabel("VehicleDest")
plt.show()
plt.scatter(df.solo_Wins,df.solo_HeadshotKills,color="blue",label="Solo")
plt.scatter(df.duo_Wins,df.duo_HeadshotKills,color="red",label="Duo")
plt.scatter(df.squad_Wins,df.squad_HeadshotKills,color="orange",label="Squad",alpha=0.5)
plt.legend()
plt.xlabel("Wins")
plt.ylabel("HS Kill")
plt.title("Scatter Plot Team Wins-Hs Kill ")
plt.show()
plt.scatter(df.solo_Wins,df.solo_VehicleDestroys,color="blue",label="Solo")
plt.scatter(df.duo_Wins,df.duo_VehicleDestroys,color="red",label="Duo")
plt.scatter(df.squad_Wins,df.squad_VehicleDestroys,color="orange",label="Squad",alpha=0.3)
plt.legend()
plt.xlabel("Wins")
plt.ylabel("VehicleDest")
plt.title("Scatter Plot Team VehicleDest-Wins")
plt.show()
plt.plot(df.solo_WalkDistance,df.solo_Kills,".",label="Solo")
plt.legend()
plt.xlabel("WalkDis")
plt.ylabel("Kill")
plt.title("Plot Solo WalkDis-KillRating")
plt.show()
plt.plot(df.duo_WalkDistance,df.duo_Kills,".",color="Red",label="Duo")
plt.legend()
plt.xlabel("WalkDis")
plt.ylabel("KillRating")
plt.title("Plot Duo WalkDis-KillRating")
plt.show()
plt.plot(df.squad_WalkDistance,df.squad_Kills,".",color="orange",label="Squad")
plt.legend()
plt.xlabel("WalkDis")
plt.ylabel("KillRating")
plt.title("Plot Squad WalkDis-KillRating")
plt.show()
plt.hist(df.solo_Top10Ratio,label="Solo-Top10")
plt.title("Histogram Plot Solo-Top10")
plt.legend()
plt.xlabel("SoloTop10")
plt.ylabel("Frequency")
plt.show()
plt.hist(df.duo_Top10Ratio,color="red",label="Duo-Top10")
plt.title("Histogram Plot Duo-Top10")
plt.legend()
plt.xlabel("DuoTop10")
plt.ylabel("Frequency")
plt.show()
plt.hist(df.squad_Top10Ratio,color="orange",label="Squad-Top10")
plt.title("Histogram Plot Squad-Top10")
plt.legend()
plt.xlabel("SquadTop10")
plt.ylabel("Frequency")
plt.show()
plt.hist(df.solo_Top10Ratio,label="Solo-Top10",bins=15)
plt.hist(df.duo_Top10Ratio,color="red",label="Duo-Top10",bins=15)
plt.hist(df.squad_Top10Ratio,color="orange",label="Squad-Top10",bins=15)
plt.title("Histogram Plot Team-Top10")
plt.legend()
plt.xlabel("TeamTop10")
plt.ylabel("Frequency")
plt.show()
mn_solo=df.solo_Heals.mean()
mn_duo=df.duo_Heals.mean()
mn_squad=df.squad_Heals.mean()
print("Heals_Mean_Solo:",mn_solo,"\n","Heals_Mean_Duo:",mn_duo,"\n","Heals_Mean_Squad:",mn_squad,sep="")
game_mood=np.array(["Solo","Duo","Squad"])
print("Game_mood",game_mood)
heals=np.array([mn_solo,mn_duo,mn_squad])
print("G_Mood_Wins",heals)
plt.bar(game_mood,heals,label="Game Mood-Heals")
plt.legend()
plt.xlabel("GAME MOOD")
plt.ylabel("HEALS")
plt.title("Bar Plot GameMood-Heals")
plt.show()
longest_s=list(df["solo_LongestKill"])
max_s=longest_s[0]
for i in longest_s:
    if max_s>i:
        continue
    else:
        max_s=i
max_s
data={"player_name":list(df.player_name),
     "solo_LongestKill":list(df.solo_LongestKill),
      "solo_Wins":list(df.solo_Wins)}
dataF=pd.DataFrame(data)
dataF.head()
kill_s=dataF["solo_LongestKill"]==max_s
dataF[kill_s]
longest_d=list(df["duo_LongestKill"])
max_d=longest_d[0]
for j in longest_d:
    if max_d>j:
        continue
    else:
        max_d=j
max_d
data1={"player_name":list(df.player_name),
     "duo_LongestKill":list(df.duo_LongestKill),
      "duo_Wins":list(df.duo_Wins)}
dataF1=pd.DataFrame(data1)
dataF1.head()
kill_d=dataF1["duo_LongestKill"]==max_d
dataF1[kill_d]
longest_sq=list(df["squad_LongestKill"])
max_sq=longest_sq[0]
for k in longest_sq:
    if max_sq>k:
        continue
    else:
        max_sq=k
max_sq
data2={"player_name":list(df.player_name),
     "squad_LongestKill":list(df.squad_LongestKill),
      "squad_Wins":list(df.squad_Wins)}
dataF2=pd.DataFrame(data2)
dataF2.head()
kill_sq=dataF2["squad_LongestKill"]==max_sq
dataF2[kill_sq]
df["total_MostSurvivalTime"]=df["solo_MostSurvivalTime"]+df["duo_MostSurvivalTime"]+df["squad_MostSurvivalTime"]
df["total_Wins"]=df["solo_Wins"]+df["duo_Wins"]+df["squad_Wins"]
df.columns
wins_t=list(df["total_Wins"])
max_t=wins_t[0]
for l in wins_t:
    if max_t>l:
        continue
    else:
        max_t=l
max_t
data_t={"player_name":list(df.player_name),
        "total_MostSurvivalTime":list(df.total_MostSurvivalTime),
      "total_Wins":list(df.total_Wins)}
dataFrame=pd.DataFrame(data_t)
kill_t=dataFrame["total_Wins"]==max_t
dataFrame[kill_t]