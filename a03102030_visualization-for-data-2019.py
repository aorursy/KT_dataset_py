import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os

import seaborn as sns

import math

import datetime
os.listdir("../input/pga-tour-20102018-data/")
data_2019=pd.read_csv("../input/pga-tour-20102018-data/2019_data.csv")

data_2019.head()
len(data_2019)
data_2019.isnull().sum()
data_2019['year']=data_2019.Date.apply(lambda x:x.split('-')[0])

data_2019['month']=data_2019.Date.apply(lambda x:x.split('-')[1])

data_2019['day']=data_2019.Date.apply(lambda x:x.split('-')[2])

data_2019.head()
data_2019.Statistic.unique()
type_stat=pd.DataFrame({'type':data_2019.Statistic.value_counts().sort_values(ascending=False)[:10].index,'value':data_2019.Statistic.value_counts().sort_values(ascending=False)[:10].values})

type_stat


qx = type_stat.plot(kind='pie', figsize=(10,7), y='value', labels=list(type_stat.type),autopct='%1.1f%%', pctdistance=0.9, radius=1.2)

plt.legend(loc=0, bbox_to_anchor=(1.10,1.0)) 

plt.title('Top 10 Statistic by number', weight='bold', size=14,y=1.08)

plt.axis('equal')

plt.ylabel('')

plt.show()

plt.clf()

plt.close()
DD=data_2019[data_2019.Variable=='Driving Distance - (TOTAL DISTANCE)'].iloc[:,[0,4]]

DD['Value_new']=DD['Value'].apply(lambda x:''.join(x.split(',')))

DD.Value_new=DD.Value_new.astype(int)

DD=DD.drop(columns='Value')

plt.figure(figsize=(22,15))

plt.subplot(2,2,1)

DD.groupby('Player Name')['Value_new'].max().sort_values(ascending=False)[:10].sort_values().plot.barh(color='darkorange')

plt.title("Top 10 player by Total Driving Distance",size=20)

plt.xlabel('meters')

plt.ylabel('')

plt.subplot(2,2,2)

DD=data_2019[data_2019.Variable=='Driving Distance - (AVG.)'].iloc[:,[0,4]]

DD.Value=DD.Value.astype(float)

DD.groupby('Player Name')['Value'].max().sort_values(ascending=False)[:10].sort_values().plot.barh(color='navy')

plt.title("Top 10 player by Average Driving Distance",size=20)

plt.xlabel('meters')

plt.ylabel('')

plt.subplot(2,2,3)

DD=data_2019[data_2019.Variable=='Driving Distance - (ROUNDS)'].iloc[:,[0,4]]

DD.Value=DD.Value.astype(float)

DD.groupby('Player Name')['Value'].max().sort_values(ascending=False)[:10].sort_values().plot.barh(color='purple')

plt.title("Top 10 player by Rounds Driving Distance",size=20)

plt.xlabel('meters')

plt.ylabel('')

plt.subplot(2,2,4)

DD=data_2019[data_2019.Variable=='Driving Distance - (TOTAL DRIVES)'].iloc[:,[0,4]]

DD.Value=DD.Value.astype(float)

DD.groupby('Player Name')['Value'].max().sort_values(ascending=False)[:10].sort_values().plot.barh()

plt.title("Top 10 player by TOTAL DRIVES Driving Distance",size=20)

plt.xlabel('meters')

plt.ylabel('')

DD=data_2019[data_2019.Variable=='Driving Distance - (TOTAL DISTANCE)'].iloc[:,[0,4,6]]

DD['Value_new']=DD['Value'].apply(lambda x:''.join(x.split(',')))

DD.Value_new=DD.Value_new.astype(int)

DD=DD.drop(columns='Value')

plt.figure(figsize=(22,22))

for i in range(len(DD.month.unique())):

    DD_month=DD[DD.month==DD.month.unique()[i]]

    plt.subplot(4,2,i+1)

    DD_month.groupby('Player Name')['Value_new'].max().sort_values(ascending=False)[:10].sort_values().plot(kind='barh')

    plt.title("Top 10 player by Total Driving Distance in month"+DD.month.unique()[i],size=20)

    plt.xlabel('meters')

    plt.ylabel('')
TE=data_2019[data_2019.Variable=='Total Eagles - (ROUNDS)'].iloc[:,[0,4]]

plt.figure(figsize=(22,15))



plt.subplot(1,2,1)

TE.Value=TE.Value.astype(int)

TE.groupby('Player Name')['Value'].max().sort_values(ascending=False)[:10].sort_values().plot.barh(color='firebrick')

plt.title("Top 10 player by Total Eagles(Rounds)",size=20)

plt.xlabel('counts')

plt.ylabel('')



plt.subplot(1,2,2)

TE=data_2019[data_2019.Variable=='Total Eagles - (TOTAL)'].iloc[:,[0,4]]

TE=TE.dropna(subset=['Value'])

TE.Value=TE.Value.astype(int)

TE.groupby('Player Name')['Value'].max().sort_values(ascending=False)[:10].sort_values().plot.barh(color='cyan')

plt.xticks(np.linspace(0, 20, 5))

plt.title("Top 10 player by Total Eagles(Total)",size=20)

plt.xlabel('counts')

plt.ylabel('')



def TOTAvgRank_DD_DAP_TE_TB(player,month):

    Ranktype=['Driving Distance - (TOTAL DRIVES)','Driving Accuracy Percentage - (%)','Total Eagles - (TOTAL)','Total Birdies - (TOTAL)']

    B={}

    for i in range(len(Ranktype)):

        A=data_2019[data_2019.Variable==Ranktype[i]]

        A=A[A.month==month]

        A.dropna(subset=['Value'])

        A.Value=A.Value.astype(float)

        value=A.groupby(['Player Name'])['Value'].agg('mean')

        rank=A.groupby(['Player Name'])['Value'].agg('mean').rank(method='min',ascending=False)

        new=pd.DataFrame({'rank':rank,'value':value})

        B['type'+str(i)]=str(Ranktype[i])

        B['rank'+str(i)]=str(new[new.index==player].iloc[0,0])+"/"+str(max(rank))

        B['value'+str(i)]=str(new[new.index==player].iloc[0,1])



    return B

plt.figure(figsize=(25,35))

j=1

for k in ['01','02','03','04','05','06','07','08']:

    Aaron=TOTAvgRank_DD_DAP_TE_TB('Aaron Baddeley',str(k))

    y=[]

    x=[]

    n=[]

    for i in range(4):

        r1,r2=Aaron['rank'+str(i)].split('/')

        R=float(r1)/float(r2)

        R=1-R

        y.append(1.5+R*math.sin(math.pi/4+i*math.pi/2))

        x.append(1.5+R*math.cos(math.pi/4+i*math.pi/2))

        n.append(Aaron['type'+str(i)]+" rank: "+Aaron['rank'+str(i)])



    x.append(x[0])

    y.append(y[0])

    plt.subplot(4,2,j)

    plt.plot(x,y, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2)

    for i, txt in enumerate(n):

        plt.annotate(txt, (x[i]-0.2, y[i]))

    plt.xlim(0.5,2.5)

    plt.ylim(0.5,2.5)

    plt.fill(x, y,"coral")

    plt.plot( 1.5, 1.5, marker='o', markerfacecolor='blue', markersize=8, linewidth=2)

    plt.title("Aaron Baddeley's performance in month "+str(k),size=18)

    j+=1
plt.figure(figsize=(25,35))

j=1

for k in ['01','02','03','04','05','06','07','08']:

    Sungjae=TOTAvgRank_DD_DAP_TE_TB('Sungjae Im',str(k))

    y=[]

    x=[]

    n=[]

    for i in range(4):

        r1,r2=Sungjae['rank'+str(i)].split('/')

        R=float(r1)/float(r2)

        R=1-R

        y.append(1.5+R*math.sin(math.pi/4+i*math.pi/2))

        x.append(1.5+R*math.cos(math.pi/4+i*math.pi/2))

        n.append(Sungjae['type'+str(i)]+" rank: "+Sungjae['rank'+str(i)])



    x.append(x[0])

    y.append(y[0])

    plt.subplot(4,2,j)

    plt.plot(x,y, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2)

    for i, txt in enumerate(n):

        plt.annotate(txt, (x[i]-0.2, y[i]))

    plt.xlim(0.5,2.5)

    plt.ylim(0.5,2.5)

    plt.fill(x, y,"greenyellow")

    plt.plot( 1.5, 1.5, marker='o', markerfacecolor='blue', markersize=8, linewidth=2)

    plt.title("Sungjae Im's performance in month "+str(k),size=18)

    j+=1
data_2019[data_2019.Statistic=='Official World Golf Ranking'].Variable.unique()
OWGR_TP=data_2019[data_2019.Variable=='Official World Golf Ranking - (TOTAL POINTS)']

OWGR_TP.Value=OWGR_TP.Value.astype(float)



fig, axes = plt.subplots(1, 2,figsize=(18,8))



OWGR_TP.groupby('Player Name')['Value'].mean().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color='brown',ax=axes[0])

axes[0].set_title('Top 10 player by Average of Total Points',size=20)

axes[0].set_xlabel('points')

axes[0].set_ylabel('')



OWGE_TP10=OWGR_TP[OWGR_TP['Player Name'].isin(list(OWGR_TP.groupby('Player Name')['Value'].mean().sort_values(ascending=False)[:10].index[:10]))]

OWGE_TP10.groupby(['Player Name','Date'])['Value'].agg('mean').unstack('Player Name').plot(ax=axes[1])

axes[1].set_title('Official World Golf Ranking - (TOTAL POINTS)',size=20)

axes[1].set_ylabel('Total points')

axes[1].set_xlim([0,30])

plt.xticks([0,5,10,15,20,25,29], ['2019-01-27','2019-03-03','2019-04-07','2019-05-12','2019-06-16','2019-07-21','2019-08-18'], rotation=0)