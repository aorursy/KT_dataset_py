import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

import plotly.express as px

init_notebook_mode(connected=True)

cf.go_offline()
df1=pd.read_csv("../input/premier-league/stats.csv")
df1.head()
df2=pd.read_csv("../input/premier-league/results.csv")
df1.columns
df2.head()
df1.describe()
class Table:

    def __init__(self,data1,data2):

        self.data1=data1

        self.data2=data2

    def dictionary(self,season):

        teams=list(self.data1[self.data1["season"]==season]["team"])

        dict1={}

        for i in teams:

            team=i

            

            W=int(list(self.data1[(self.data1["team"]==team)&(self.data1["season"]==season)]["wins"])[0])

            L=int(list(self.data1[(self.data1["team"]==team)&(self.data1["season"]==season)]["losses"])[0])

            D=list((self.data2[((self.data2["home_team"]==team)|(self.data2["away_team"]==team))&(self.data2["season"]==season)&(self.data2["result"]=="D")]["result"]).value_counts())[0]

            Pid=W+L+D

            GA=int((np.array(self.data2[(df2["home_team"]==team)&(self.data2["season"]==season)]["away_goals"]).sum())+(np.array(self.data2[(self.data2["away_team"]==team)&(self.data2["season"]==season)]["home_goals"]).sum()))

            GF=int((np.array(self.data2[(df2["home_team"]==team)&(self.data2["season"]==season)]["home_goals"]).sum())+(np.array(self.data2[(self.data2["away_team"]==team)&(self.data2["season"]==season)]["away_goals"]).sum()))

            GD=int(GF-GA)

            Pts=(W*3)+D

            dict1[team]={"Pid":Pid,"W":W,"D":D,"L":L,"GF":GF,"GA":GA,"GD":GD,"Pts":Pts}

        return dict1

    def Table_for_Particular_Season(self,season):

        t=self.dictionary(season)

        index=np.array(list(t.keys()))

        colmns=["Pid","W","D","L","GF","GA","GD","Pts"]

        df=pd.DataFrame(data=t.values(),index=index,columns=["Pid","W","D","L","GF","GA","GD","Pts"])

        return df

    def number_of_goals(self,season):

        #g=self.dictionary(season)

        plt.figure(figsize=(10,10))

        goals=self.data1[self.data1["season"]==season]["goals"]

        teams=self.data1[self.data1["season"]==season]["team"]

        fig=sns.barplot(y=teams,x=goals,orient='h',errwidth=0.55,saturation=.98)

    def away_wins_vs_home_wins(self,season):

        teams=list(self.data1[self.data1["season"]==season]["team"])

        dict1={}

        away_win_list=[]

        away_loss_list=[]

        home_win_list=[]

        home_loss_list=[]

        labels=[]

        for i in teams:

            liv=self.data2[((self.data2["home_team"]==i)|(self.data2["away_team"]==i))&(self.data2["season"]==season)][["home_team","away_team","home_goals","away_goals"]]

            away_loss=liv[liv["away_team"]==i][liv["home_goals"]>liv["away_goals"]].count()["home_team"]

            away_win=liv[liv["away_team"]==i][liv["home_goals"]<liv["away_goals"]].count()["home_team"]

            home_loss=liv[liv["home_team"]==i][liv["home_goals"]<liv["away_goals"]].count()["home_team"]

            home_win=liv[liv["home_team"]==i][liv["home_goals"]>liv["away_goals"]].count()["home_team"]

            away_win_list.append(away_win)

            away_loss_list.append(away_loss)

            home_win_list.append(home_win)

            home_loss_list.append(home_loss)

            labels.append(i)

            dict1[i]={"home_win":home_win,"home_loss":home_loss,"away_win":away_win,"away_loss":away_loss}

        labels=list(dict1.keys())

        fig, axs=plt.subplots(4,1,figsize=(10,20))

        sns.barplot(y=labels,x=away_win_list,orient="h",ax=axs[2],errwidth=0.55,saturation=.98)

        sns.barplot(y=labels,x=away_loss_list,orient="h",ax=axs[3],errwidth=0.55,saturation=.98)

        sns.barplot(y=labels,x=home_win_list,orient="h",ax=axs[0],errwidth=0.55,saturation=.98)

        sns.barplot(y=labels,x=home_loss_list,orient="h",ax=axs[1],errwidth=0.55,saturation=.98)

        axs[0].set_title("home_wins")

        axs[1].set_title("home_loss")

        axs[2].set_title("away_wins")

        axs[3].set_title("away_loss")

        return dict1
t=Table(df1,df2)
# you can use season=input() instaed

season="2016-2017"
df=t.Table_for_Particular_Season(season)
print("Premier league Table of {} season".format(season))

df
print("relegated teams in {} season were".format(season))

list(df[-3:].index)
print("number of goals in {} season by each team".format(season))

t.number_of_goals(season)
print("analysis of home-away wins and losses in {} season".format(season))

t.away_wins_vs_home_wins(season)
class Team:

    def __init__(self,data1,data2):

        self.data1=data1

        self.data2=data2

    def performance_throgh_years(self,team):

        season=self.data1[self.data1["team"]==team]["season"]

        wins=self.data1[self.data1["team"]==team]["wins"]

        losses=self.data1[self.data1["team"]==team]["losses"]

        plt.figure(figsize=(10,7))

        plt.grid()

        plt.plot(season,wins,'red',label="WINS",marker="o")

        plt.plot(season,losses,"yellow",label="LOSS",marker="o")

        plt.xticks(rotation=60,fontsize=14)

        plt.yticks(fontsize=14)

        plt.legend(fontsize=14)

        plt.show()

    def home_away_record(self,team):

        ars=self.data2[self.data2["away_team"]==team][["home_goals","away_goals"]]

        ars1=self.data2[self.data2["home_team"]==team][["home_goals","away_goals"]]

        away_wins=ars[ars["home_goals"]<ars["away_goals"]].count()["home_goals"]

        away_losses=ars[ars["home_goals"]>ars["away_goals"]].count()["home_goals"]

        home_wins=ars1[ars1["home_goals"]>ars1["away_goals"]].count()["home_goals"]

        home_losses=ars1[ars1["home_goals"]<ars1["away_goals"]].count()["home_goals"]

        return (away_wins,away_losses,home_wins,home_losses)

    
t1=Team(df1,df2)
#team=input()

team="Arsenal"
t1.performance_throgh_years(team)
print("FROM 2006-2018 {}".format(team))

print("home wins=",t1.home_away_record(team)[2])

print("home_losses=",t1.home_away_record(team)[3])

print("away_wins=",t1.home_away_record(team)[0])

print("away_losses=",t1.home_away_record(team)[1])
print("goals from oustide-Inside box")

m=df1[df1["team"]=="Manchester United"]["att_obox_goal"].sum()

c=df1[df1["team"]=="Chelsea"]["att_obox_goal"].sum()

a=df1[df1["team"]=="Arsenal"]["att_obox_goal"].sum()

mc=df1[df1["team"]=="Manchester City"]["att_obox_goal"].sum()

liv=df1[df1["team"]=="Liverpool"]["att_obox_goal"].sum()

tot=df1[df1["team"]=="Tottenham Hotspur"]["att_obox_goal"].sum()

m1=df1[df1["team"]=="Manchester United"]["att_ibox_goal"].sum()

c1=df1[df1["team"]=="Chelsea"]["att_ibox_goal"].sum()

a1=df1[df1["team"]=="Arsenal"]["att_ibox_goal"].sum()

mc1=df1[df1["team"]=="Manchester City"]["att_ibox_goal"].sum()

liv1=df1[df1["team"]=="Liverpool"]["att_ibox_goal"].sum()

tot1=df1[df1["team"]=="Tottenham Hotspur"]["att_ibox_goal"].sum()

data1=pd.DataFrame({"outside_box":[m,c,a,mc,liv,tot],"inside_box":[m1,c1,a1,mc1,liv1,tot1],"columns":"Man Chelsea Ars Manc Liv TOt".split()})

data1.iplot(kind="bar",x="columns",y=["outside_box","inside_box"],subplots=True,shape=(1,2),subplot_titles=True,colors=["red","blue"])
print("defensive record")

m=df1[df1["team"]=="Manchester United"]["clean_sheet"].sum()

c=df1[df1["team"]=="Chelsea"]["clean_sheet"].sum()

a=df1[df1["team"]=="Arsenal"]["clean_sheet"].sum()

mc=df1[df1["team"]=="Manchester City"]["clean_sheet"].sum()

liv=df1[df1["team"]=="Liverpool"]["clean_sheet"].sum()

tot=df1[df1["team"]=="Tottenham Hotspur"]["clean_sheet"].sum()

data1=pd.DataFrame({"defensive record":[m,c,a,mc,liv,tot],"columns":"Man Chelsea Ars Manc Liv TOt".split()})

px.pie(data1,values="defensive record",names="columns")