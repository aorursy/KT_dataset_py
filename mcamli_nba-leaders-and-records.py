#importing libraries

import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

import plotly.graph_objs as go



import os

print(os.listdir("../input"))
data1=pd.read_csv("../input/nba-players-stats/Seasons_Stats.csv")

data2=pd.read_csv("../input/nba17-18/nba.csv")

data3=pd.read_csv("../input/nba17-18/nba_extra.csv")

data4=pd.read_csv("../input/nba-players-stats/player_data.csv")

#I created and added data3 in order to complete data2's missing columns.
data1.head()
data2.tail()
#Dropping Uncommon and empty ones so that columns are equal.

data2.drop(data2.iloc[:,[19,24,0]],axis=1,inplace=True)

data1.drop(["GS","blanl","blank2"],axis=1,inplace=True)

data1.drop(data1.iloc[:,[0]],axis=1,inplace=True)

type(data1)
#Editing names on data2

new=[]

for i in range(0,len(data2.Player)):

    x=data2.Player[i].split("\\")

    new.append(x[0])

data2["Player"]=new
#Creating a new column in a specific place

data2.insert(0,"Year",2018.0)
#Concatenating datas

data2=pd.concat([data2,data3.loc[:,"FG":]],axis=1)
#Changing data types

data2.Age=data2.Age.astype(float)

for i in data3.loc[:,"FG":]:

    data2[i]=data2[i].astype(float)
#Finally we can concatenate the data that we will work on it.

data=pd.concat([data1,data2],axis=0,ignore_index=True)
#There are some mistakes in name of players.

new_names=[]

for i in data.Player:

    i=str(i)

    if (i[-1]=="*"):

        i=i[:-1]

    new_names.append(i)

data["Player"]=new_names    
#TOT is total value of season if the player plays for more than 1 team in a season.I don't need it.Therefore,I am dropping the values.

data=data[data.Tm!="TOT"]
#Content of columns

data.info()
#As we can guess,there are many players who share same name;therefore, i need to seperate them according to their first and last year in NBA.

player_name=list(data4.name) 



player_points=[]

player_assists=[]

player_rebounds=[]

player_blocks=[]





player_year_start=list(data4.year_start)

player_year_end=list(data4.year_end)



for i in range(0,len(player_name)):

    first_year=player_year_start[i]

    last_year=player_year_end[i]

    seperated_data=data[(data.Player==player_name[i]) & (data.Year>=first_year) & (data.Year<=last_year)]  

    player_points.append(seperated_data.PTS.sum())

    player_assists.append(seperated_data.AST.sum())

    player_rebounds.append(seperated_data.TRB.sum())

    player_blocks.append(seperated_data.BLK.sum())

    



#correlation map

f,ax=plt.subplots(figsize=(25, 25))

sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)

plt.show()
#Visualization 

dictionary={"Player":player_name,"Points":player_points}

playerpoints=pd.DataFrame(dictionary)

playerpoints=playerpoints.sort_values("Points",ascending=False)



pointsforbar=playerpoints.head(20)



plt.figure(figsize=(20,13))

sns.barplot(x=pointsforbar.Player,y=pointsforbar.Points)

plt.xticks(rotation=60)

plt.ylabel("Points")

plt.xlabel("Player Name")

plt.title("NBA All Times Points Leaders")

plt.show()
#Visualization 

dictionary={"Player":player_name,"Assists":player_assists}

playerassists=pd.DataFrame(dictionary)

playerassists=playerassists.sort_values("Assists",ascending=False)



assistforbar=playerassists.head(20)



plt.figure(figsize=(18,12))

sns.barplot(x=assistforbar.Player, y=assistforbar.Assists,palette =sns.cubehelix_palette(len(assistforbar)))

plt.xticks(rotation=60)

plt.xlabel("Player Names")

plt.ylabel("Assists")

plt.title("NBA All-Time Assists Leaders")

plt.show()
#Visualization 

dictionary={"Player":player_name,"Rebounds":player_rebounds}

playerrebounds=pd.DataFrame(dictionary)

playerrebounds=playerrebounds.sort_values("Rebounds",ascending=False)



reboundsforbar=playerrebounds.head(20)



plt.figure(figsize=(20,13))

sns.barplot(x=reboundsforbar.Player,y=reboundsforbar.Rebounds,palette=sns.cubehelix_palette(len(reboundsforbar),rot=-.5))

plt.xticks(rotation=60)

plt.ylabel("Rebounds")

plt.xlabel("Player Name")

plt.title("NBA All Times Rebounds Leaders")

plt.show()
#Visualization 

dictionary={"Player":player_name,"Blocks":player_blocks}

playerblocks=pd.DataFrame(dictionary)

playerblocks=playerblocks.sort_values("Blocks",ascending=False)



blocksforbar=playerblocks.head(20)



plt.figure(figsize=(18,12))

sns.barplot(x=blocksforbar.Player, y=blocksforbar.Blocks,palette =sns.color_palette("Reds_d",len(blocksforbar)))

plt.xticks(rotation=60)

plt.xlabel("Player Names")

plt.ylabel("Blocks")

plt.title("NBA All-Time Blocks Leaders")

plt.show()
#Creating data for visualization

year=[]

totalpoints=[]

for i in data.Year.unique():

    year.append(i)

    total=0

    x=data[data.Year==i]

    for j in x.PTS:

        total+=int(j)

    totalpoints.append(total)  

data_2=pd.DataFrame({"Year":year,"Point":totalpoints})    
#Visualization

plt.subplots(figsize =(20,10))

sns.pointplot(x="Year",y="Point",data=data_2,color="red",alpha=1.2)

plt.xlabel("Years",fontsize = 25)

plt.ylabel("Total Points",fontsize = 25)

plt.xticks(rotation=70)

plt.title("Total Points According To Years",fontsize = 30,color='blue')

plt.grid()
#Creating data for visualization

player_3ptp=[]

made3pt=[]

for i in range(0,len(player_name)):

    x=0

    y=0

    first_year=player_year_start[i]

    last_year=player_year_end[i]

    x+=data[(data.Player==player_name[i]) & (data.Year>=first_year) & (data.Year<=last_year)]["3P"].sum()   

    y+=data[(data.Player==player_name[i]) & (data.Year>=first_year) & (data.Year<=last_year)]["3PA"].sum()   

    player_3ptp.append(x/y)

    made3pt.append(x)

    

player3pt=pd.DataFrame({"Player":player_name,"3P%":player_3ptp,"3P":made3pt})

player3pt=player3pt[player3pt["3P"]>250]

player3pt=player3pt.sort_values("3P%",ascending=False)
#Visualization

first_15=player3pt.head(15)

plt.figure(figsize=(15,16))

plt.bar(first_15.Player,first_15["3P%"])

plt.xticks(rotation=65)

plt.xlabel("Player Names")

plt.ylabel("3P%")

plt.title("3-Pt Field Goal Percentage")

plt.ylim(0,0.5)

plt.grid()

plt.show()
#Creating data for visualization

player_fg=[]

fieldgoaltotal=[]

for i in range(0,len(player_name)):

    x=0

    y=0

    first_year=player_year_start[i]

    last_year=player_year_end[i]

    x+=data[(data.Player==player_name[i]) & (data.Year>=first_year) & (data.Year<=last_year)]["FG"].sum()   

    y+=data[(data.Player==player_name[i]) & (data.Year>=first_year) & (data.Year<=last_year)]["FGA"].sum()   

    player_fg.append(x/y)

    fieldgoaltotal.append(x)

    

playerfgp=pd.DataFrame({"Player":player_name,"FG%":player_fg,"FG":fieldgoaltotal})

playerfgp=playerfgp[playerfgp["FG"]>2000]

playerfgp=playerfgp.sort_values("FG%",ascending=False)
#Visualization 

fgp_bar=playerfgp.head(20)

plt.figure(figsize=(18,12))

sns.barplot(x=fgp_bar.Player, y=fgp_bar["FG%"],palette =sns.cubehelix_palette(len(fgp_bar)))

plt.xticks(rotation=60)

plt.xlabel("Player Names")

plt.ylabel("FG%")

plt.title("NBA Leaders for Field Goal Percentage")
#Detecting how many teams are there in NBA history.

data.Tm.unique()
#Adding datas whether teams play in East or West

conferences={"NYK":"East","PHW":"East","ROC":"East","BLB":"East","SYR":"East","WSC":"East","BOS":"East","PHI":"East","CLE":"East",

             "BUF":"East","CAP":"East","NOJ":"East","WSB":"East","NYN":"East","NJN":"East","CHH":"East","MIA":"East",

             "ORL":"East","TOR":"East","WAS":"East","CHA":"East","BRK":"East","NOP":"East","CHO":"East",

             "FTW":"West","INO":"West","CHS":"West","DNN":"West","TRI":"West","AND":"West","WAT":"West",

             "SHE":"West","MNL":"West","STB":"West","MLH":"West","STL":"West","DET":"West","CIN":"West",

             "LAL":"West","CHP":"West","SFW":"West","CHZ":"West","BAL":"West","SDR":"West","SEA":"West","PHO":"West","POR":"West","KCO":"West",

             "KCK":"West","DEN":"West","SDC":"West","UTA":"West","DAL":"West","LAC":"West","SAC":"West","MIN":"West","VAN":"West","MEM":"West",

             "OKC":"West","GSW":"West"}
#Creating new column.

data["Conferences"]=data["Tm"].map(conferences)
#There are some NBA teams that have played in east and west conference.



#For Chicago Bulls

for i in data[(data.Tm=="CHI") & (data.Year>=1981)].index:

    data.set_value(i,"Conferences","East")

for i in data[(data.Tm=="CHI") & (data.Year<1981)].index:

    data.set_value(i,"Conferences","West")

#For Milwaukee Bucks

for i in data[(data.Tm=="MIL") & (data.Year>=1981)].index:

    data.set_value(i,"Conferences","East")

for i in data[(data.Tm=="MIL") & (data.Year<1981)].index:

    data.set_value(i,"Conferences","West")

#For Atlanta Hawks

for i in data[(data.Tm=="ATL") & (data.Year>=1971)].index:

    data.set_value(i,"Conferences","East")

for i in data[(data.Tm=="ATL") & (data.Year<1971)].index:

    data.set_value(i,"Conferences","West")

#For Houston Rockets

for i in data[(data.Tm=="HOU") & (data.Year==1972)].index:

    data.set_value(i,"Conferences","West")

for i in data[(data.Tm=="HOU") & (data.Year>=1981)].index:

    data.set_value(i,"Conferences","West")

for i in data[(data.Tm=="HOU") & (data.Year<1981) & (data.Year>1972)].index:

    data.set_value(i,"Conferences","East")

#For Indiana Pacers

for i in data[(data.Tm=="IND") & (data.Year>=1980)].index:

    data.set_value(i,"Conferences","East")

for i in data[(data.Tm=="IND") & (data.Year<1980)].index:

    data.set_value(i,"Conferences","West")

#For San Antonio Spurs

for i in data[(data.Tm=="SAS") & (data.Year>=1981)].index:

    data.set_value(i,"Conferences","West")

for i in data[(data.Tm=="SAS") & (data.Year<1981)].index:

    data.set_value(i,"Conferences","East")

#For New Orleans Hornets

for i in data[(data.Tm=="NOH") & (data.Year>=2005)].index:

    data.set_value(i,"Conferences","West")

for i in data[(data.Tm=="NOH") & (data.Year<2005)].index:

    data.set_value(i,"Conferences","East")

#Creating data for visualization

dataforsub2=data[data.Conferences=="West"].groupby("Year").sum()

dataforsub=data[data.Conferences=="East"].groupby("Year").sum()
#Visualization

plt.subplots(figsize =(20,10))

plot1=sns.pointplot(x=dataforsub.index,y="PTS",data=dataforsub,color="blue",alpha=1.0,label="East")

plot2=sns.pointplot(x=dataforsub2.index,y="PTS",data=dataforsub2,color="red",alpha=1.0,label="West")



plt.xlabel("Years",fontsize = 25)

plt.ylabel("Total Points",fontsize = 25)

plt.xticks(rotation=70)

plt.title("Total Points According To Conferences",fontsize = 30,color='blue')

plt.text(55,42000,"East",color="blue",fontsize =20)

plt.text(55,32000,"West",color="red",fontsize=20)

plt.grid()