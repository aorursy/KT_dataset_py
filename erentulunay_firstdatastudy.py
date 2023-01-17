# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/PUBG_Player_Statistics.csv")
print(df.columns.values)
df.head()

x=list(df.loc[:,'duo_KillDeathRatio':'duo_DBNOs'])
df.drop(x,axis=1,inplace=True)
print(df.columns.values)                                                      #####################delete columns
df.drop(['solo_RoadKillsPg','solo_Heals','solo_Revives','squad_HealsPg','squad_VehicleDestroys','squad_WalkDistance','squad_RideDistance','squad_MoveDistance','squad_AvgWalkDistance','squad_AvgRideDistance','squad_RoadKillsPg','squad_MoveDistancePg','squad_RevivesPg'],axis=1)
df.drop(columns=['solo_Heals','squad_Heals','squad_Revives', 'squad_Boosts','squad_HealsPg','squad_WalkDistance', 'squad_RideDistance','squad_MoveDistance', 'squad_AvgWalkDistance',"squad_AvgRideDistance","squad_DBNOs","solo_WeaponAcquired","solo_VehicleDestroys","squad_RoadKillsPg","squad_MoveDistancePg","squad_RevivesPg","solo_MoveDistancePg","solo_RevivesPg","solo_RoadKillsPg","squad_DamageDealt","squad_WeaponAcquired","squad_RoadKills","squad_VehicleDestroys","solo_LongestTimeSurvived","solo_MostSurvivalTime","solo_TimeSurvivedPg","solo_Revives","solo_Boosts","solo_DamageDealt","solo_DBNOs","squad_LongestTimeSurvived","squad_MostSurvivalTime","squad_AvgSurvivalTime","squad_Suicides","squad_TimeSurvivedPg","squad_TimeSurvived","solo_TimeSurvived","solo_DamagePg","solo_Assists","solo_TeamKills","solo_DailyKills","solo_WeeklyKills","squad_TimeSurvived","squad_TeamKills","squad_DailyKills","squad_WeeklyKills"],inplace=True)
df.drop(columns=["squad_HeadshotKillRatio","squad_HeadshotKills","squad_TeamKillsPg","squad_Assists","squad_BestRating","solo_WinTop10Ratio","solo_BestRating","solo_TeamKillsPg","squad_HeadshotKillsPg"],inplace=True)
df.drop(["tracker_id","solo_AvgSurvivalTime",'solo_HeadshotKillsPg',"solo_HealsPg",'solo_Suicides','solo_RoadKills','solo_WalkDistance','solo_RideDistance','solo_MoveDistance','solo_AvgWalkDistance','solo_AvgRideDistance'],axis=1,inplace=True)
df.drop(["solo_HeadshotKillRatio","solo_Top10Ratio","solo_Days"],axis=1,inplace=True)      #####################delete columns
df=df.loc[:,:"squad_KillDeathRatio"]
x=list(df.columns)
df= df[[each for each in x if "Pg"  not in each]] ##################################delete columns
df.drop(["squad_KillDeathRatio"],axis=1,inplace=True)
df= df[(df["solo_WinRatio"]>10) & (df["solo_Wins"]>25)]
df.reset_index(drop=True,inplace= True)
df.rename(columns={"player_name":"player"},inplace=True)   #### change column name
df.corr()
f,ax= plt.subplots(figsize= (18,18))
sns.heatmap(df.corr(),annot=True,fmt=".1f",linewidths=.3,ax=ax)
plt.show()
df.solo_Wins.plot(kind="line",grid=True,label="solo_Wins",color="g",alpha=0.5)
df.solo_WinRatio.plot(kind="line",grid=True,label="WinRatio",color="r",alpha=0.5,linestyle=":")
plt.legend(loc="upper right")
plt.show()
#use line plot when you need to examine data with time
df.plot(kind="scatter",x="solo_Losses",y="solo_WinRatio",grid=True,alpha=0.7,color="r")
xlabel=("solo_Losses")
ylabel=("solo_WinRatio")
plt.show()
#use scatter when you need to correlation
df.solo_Wins.plot(kind="hist",bins=20,figsize=(15,15))
plt.show()
##### frequence if you need
for index,value in df[["solo_Wins"]][0:20].iterrows():
    print(index,value)    
    ###### you can get index and value same time
df.shape    #### how many index and column you have
df.describe()
liste= list(df.solo_LongestKill)
liste.sort()
print(liste)          ######list sort method you can't equal liste.sort() to something
df.solo_LongestKill.sort_values()  #####series sort method
df[["solo_LongestKill"]].sort_values(by="solo_LongestKill")  ##### dataframe sort method
df.solo_MaxKillStreaks.value_counts(dropna=False) ####count values
df.boxplot(["solo_Wins"]) ###### BOXPLOT is really good for outlier 
new_df = df.head()
melted =pd.melt(frame=new_df,id_vars=("player"),value_vars=("solo_Wins","solo_Losses")) #####we used fro seaborn graphics
print(melted)
melted.pivot(index="player",columns="variable",values="value")
x=df.head()
y=df.tail()
a=pd.concat([x,y],axis=0)
print(a)          #####concatenate two data types
df.solo_KillDeathRatio= df.solo_KillDeathRatio.astype(int)
data1=df.copy()
data1.info()
assert 1==1
assert df.solo_HeadshotKills.notnull().all()
df_list=df.values
print(df_list)
######studying dictionary dict
dictionary={"isim":("Ali","Veli","kenan", "ahmet"),"yas":(20,21,15,17),"maas":(1500,2000,500,8000)}
dictionary["isim"]="ad"
dictionary["emekli"]=(50,55,60,65)
print(dictionary)
def f(*args):
    for i in args:
        print(i)
f(1)
###### how to zip code try to understand
x=["ali","veli","49","50"]
y=[["ahmet"],["mehmet"],[60],[31]]
a= list(zip(x,y))
b=dict(a)
print(b)
x=pd.DataFrame(b)
print(x) 
######## list to dataframe
country = ["Spain","France"]
population = ["11","12"]
list_col = [country,population]
list_label = ["country","population"]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
dataframe = pd.DataFrame(data_dict)
dataframe["country"]=["Ukraine","ABD"]
print(dataframe)
###### working with datetime
df_head=df.head()
date_time=["2010-05-02","2010-05-03","2015-08-05","2018-06-05","2018-06-07"]
date=pd.to_datetime(date_time)
df_head["date"]=date
df_head.set_index("date",inplace=True)
print(df_head)
######resample values
#df_head.resample(rule="A").mean()
df_head.resample(rule="A").mean().interpolate("linear")
#df_head.resample(rule="M").mean()

df.player_name[df.solo_Wins>100]
df["yeni"]= df.solo_Wins - df.solo_Losses
df.yeni.apply(lambda n:n/2)
data1=df.copy()
data1.index.name= "index"
data1= data1.set_index("player")
data1.loc["TwitchTV_Gudybay"]
