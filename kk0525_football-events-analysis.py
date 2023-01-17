import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import scipy as sp

pd.set_option('display.width',800)

pd.set_option('display.max_columns',1500)



events=pd.read_csv('../input/events.csv')

events.head()
events.isnull().sum()
encoding=pd.read_table('../input/dictionary.txt',delim_whitespace=False,names=('num','events'));encoding
event_types = {1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded'}

event_types2 = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal'}

sides = {1:'Home', 2:'Away'}

shot_places = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner'}

shot_outcomes = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar'}

locations = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded'}

bodyparts = {1:'right foot', 2:'left foot', 3:'head'}

assist_methods = {0:np.nan, 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball'}

situations = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick'}

goal=events[events.is_goal==1] # select games in which get the score

goal['event_type']=goal.event_type.map(event_types)

goal['event_type2']=goal.event_type2.map(event_types2)

goal['side'] = goal['side'].map(sides)

goal['shot_place'] =goal['shot_place'].map(shot_places)

goal['shot_outcome'] = goal['shot_outcome'].map(shot_outcomes)

goal['location'] = goal['location'].map(locations)

goal['bodypart'] = goal['bodypart'].map(bodyparts)

goal['assist_method'] =goal['assist_method'].map(assist_methods)

goal['situation'] = goal['situation'].map(situations)

goal.head()
plt.rc("xtick",labelsize=20)

plt.rc("ytick",labelsize=20)

fig=plt.figure(figsize=(13,10))

plt.hist(goal.time,bins=100,color='blue',width=1)

plt.xlabel("TIME",fontsize=20)

plt.ylabel("goal counts",fontsize=20)

plt.title("goal counts vs time",fontsize=25)

x=goal.groupby(by='time')['time'].count().sort_values(ascending=False).index[0]

y=goal.groupby(by='time')['time'].count().sort_values(ascending=False).iloc[0]

x1=goal.groupby(by='time')['time'].count().sort_values(ascending=False).index[1]

y1=goal.groupby(by='time')['time'].count().sort_values(ascending=False).iloc[1]

plt.text(x=x-10,y=y+10,s='time:'+str(x)+',max:'+str(y),fontsize=15,fontdict={'color':'red'})

plt.text(x=x1-10,y=y1+10,s='time:'+str(x1)+',the 2nd max:'+str(y1),fontsize=15,fontdict={'color':'black'})

plt.show() 
fig=plt.figure(figsize=(15,10))

plt.grid()



plt.rc('xtick',labelsize=20)

plt.rc('ytick',labelsize=20)

plt.hist(goal[goal.side=='Home'].time,bins=100,label="Home team",color='pink',width=1)

plt.hist(goal[goal.side=='Away'].time,bins=100,label="Away team",width=1)



plt.xlabel("Time",fontsize=15)

plt.ylabel("counts",fontsize=15)

plt.title("goal counts vs time by side",fontsize=20)

plt.legend(loc="upper left",fontsize=20)

plt.show()
goal1=goal.copy()

goal1.loc['bodypart']=goal1['bodypart'].fillna('right foot')

goal1.time=goal1['time'].fillna(goal1.time.median())
plt.figure(figsize=(10,8))

data1=goal1.groupby(by=['bodypart'])['bodypart'].count()

colors=["cyan","grey","pink"]

plt.pie(data1,colors=colors,autopct='%1.1f%%',labels=['head','left foot','right foot'],startangle=90)

plt.axis('equal')

plt.title("Percentage of bodyparts for goals",fontsize=17)

plt.legend(fontsize=12,loc='upper right')

plt.show()
plt.figure(figsize=(10,8))

plt.hist(goal1[goal1["bodypart"]=='right foot']["time"],width=1,bins=100,color="green",label="Right foot",alpha=0.6)

plt.hist(goal1[goal1["bodypart"]=='left foot']["time"],width=1,bins=100,color="grey",label="Left foot") 

plt.hist(goal1[goal1["bodypart"]=='head']["time"],width=1,bins=100,color="pink",label="Headers") 

 

plt.xlabel("Minutes",fontsize=20)

plt.ylabel("Number of goals",fontsize=20)

plt.legend(loc='upper left',fontsize=15)

plt.title("Number of goals (by body parts) against Time during match",fontsize=20,fontweight="bold")

plt.show()
set(goal1.situation)
goal1.situation.isnull().sum();

goal1.situation.isnull().any();

goal1['situation']=goal1.situation.fillna('Open play')
plt.figure(figsize=(10,10))

size=[goal1[goal1.situation=='Open play'].shape[0],goal1[goal1.situation=='Set piece'].shape[0],goal1[goal1.situation=='Corner'].shape[0],goal1[goal1.situation=='Free kick'].shape[0]]

colors=['pink','purple','blue','green']

plt.pie(size,colors=colors,autopct='%.1f%%',textprops={"fontsize":15},labels=['Open play','Set piece','Corner','Free kick'])

plt.title('Percentage of each situation for goals',fontsize=25)

plt.axis('equal')

plt.show()
plt.figure(figsize=(20,10))

plt.hist(goal1[goal1.situation=='Open play'].time,bins=100,width=1,color='red',label='Open play')

plt.hist(goal1[goal1.situation=='Set piece'].time,bins=100,width=1,color='purple',label='Set piece')

plt.hist(goal1[goal1.situation=='Corner'].time,bins=100,width=1,color='blue',label='Corner')

plt.hist(goal1[goal1.situation=='Free kick'].time,bins=100,width=1,color='green',label='Free kick')

plt.xlabel("TIME",fontsize=25)

plt.ylabel("goal counts",fontsize=25)

plt.rc('xtick',labelsize=15)

plt.rc('ytick',labelsize=15)

plt.title('Percentage of each situation for goals',fontsize=25,fontweight='bold')

plt.legend(loc='upper left',fontsize=15)

plt.show()
goal1.location.isnull().sum()

set(goal1.location)
goal1.replace({np.nan:"None"},inplace=True)

goal1.groupby(by='assist_method')['time'].count()

plt.figure(figsize=(8,8))

plt.pie(goal1.groupby(by='assist_method')['time'].count(),labels=set(list(goal1.assist_method)),autopct='%1.2f%%',textprops={'fontsize':15,'color':'black'},startangle=30)

plt.title("assist_method info",fontsize=20)

plt.axis('equal')

plt.show()
goal1.head()