import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
event=pd.read_csv("../input/athlete_events.csv")
event.head()
rgn=pd.read_csv("../input/noc_regions.csv")
rgn.head()
#rgn.NOC.unique
event[['Age','Height','Weight']].describe()
#print(event[["Games","Year","Season"]])
event.sort_values('Games',inplace=True)
event.Games.unique()
event[event['Season']=='Summer'].Sport.unique()
event[event['Season']=='Winter'].Sport.unique()
fig=plt.figure(figsize=(20,20))
plt.xticks(rotation=90)
ax=fig.add_subplot(111)
ax.plot(event["M" == event['Sex']]['Games'],event["M" == event['Sex']]['Age'],'bo')
ax.plot(event["F" == event['Sex']]['Games'],event["F" == event['Sex']]['Age'],'rx')
plt.xlabel('Games')
plt.ylabel('Age')
plt.legend({'Male','Female'})
plt.show()
year_count_M={}
for y in event.Games.unique():
    year_count_M[y]=event[event.Sex=='M'][event.Games==y]['Season'].count()
year_count_F={}
for y in event.Games.unique():
    year_count_F[y]=event[event.Sex=='F'][event.Games==y]['Season'].count()
year_count_F=sorted(year_count_F.items())
year_count_M=sorted(year_count_M.items())

fig=plt.figure(figsize=(15,10))
plt.xticks(rotation=90)
ax=fig.add_subplot(111)

x,y=zip(*year_count_M)
ax.bar(x,y,color='blue',width=0.3,align='center')

x,y=zip(*year_count_F)
ax.bar(x,y,color='red',width=0.3,align='edge')

plt.ylabel('number of participants')
plt.xlabel('Games')
plt.legend({'Male','Female'})
#plt.show()
year_count_M_G={}
for y in event.Games.unique():
    year_count_M_G[y]=event[event.Sex=='M'][event.Games==y][event.Medal=='Gold']['Season'].count()
year_count_M_S={}
for y in event.Games.unique():
    year_count_M_S[y]=event[event.Sex=='M'][event.Games==y][event.Medal=='Silver']['Season'].count()
year_count_M_B={}
for y in event.Games.unique():
    year_count_M_B[y]=event[event.Sex=='M'][event.Games==y][event.Medal=='Bronze']['Season'].count()
    
year_count_M_G=sorted(year_count_M_G.items())
year_count_M_S=sorted(year_count_M_S.items())
year_count_M_B=sorted(year_count_M_B.items())

fig=plt.figure(figsize=(15,10))
plt.xticks(rotation=90)
ax=fig.add_subplot(111)

xg,yg = zip(*year_count_M_G)
ax.bar(xg,yg,color='gold',width=0.5)
#print(year_count_M_G)

xs,ys = zip(*year_count_M_S)
ax.bar(xs,ys,bottom=yg,color='silver',width=0.5)
#print(year_count_M_S)

xb,yb = zip(*year_count_M_B)
ax.bar(xb,yb,bottom=np.array(ys)+np.array(yg),color='brown',width=0.5)
#print(year_count_M_B)

plt.ylabel('Medal count')
plt.xlabel('Games')
plt.legend({'Bronze','Silver','Gold'})
plt.title('male')
plt.show()
year_count_F_G={}
for y in event.Games.unique():
    year_count_F_G[y]=event[event.Sex=='F'][event.Games==y][event.Medal=='Gold']['Season'].count()
year_count_F_S={}
for y in event.Games.unique():
    year_count_F_S[y]=event[event.Sex=='F'][event.Games==y][event.Medal=='Silver']['Season'].count()
year_count_F_B={}
for y in event.Games.unique():
    year_count_F_B[y]=event[event.Sex=='F'][event.Games==y][event.Medal=='Bronze']['Season'].count()

year_count_F_G=sorted(year_count_F_G.items())
year_count_F_S=sorted(year_count_F_S.items())
year_count_F_B=sorted(year_count_F_B.items())

fig=plt.figure(figsize=(15,10))
plt.xticks(rotation=90)
ax=fig.add_subplot(111)

xg,yg = zip(*year_count_F_G)
ax.bar(xg,yg,color='gold',width=0.5)

xs,ys = zip(*year_count_F_S)
ax.bar(xs,ys,bottom=yg,color='silver',width=0.5)

xb,yb = zip(*year_count_F_B)
ax.bar(xb,yb,bottom=np.array(ys)+np.array(yg),color='brown',width=0.5)

plt.ylabel('Medal count')
plt.xlabel('Games')
plt.legend({'Bronze','Silver','Gold'})
plt.title('female')
plt.show()
#event.Medal.unique()
len(event.Team.unique())
plt.figure(figsize=(20,20))
tmp=event[(event.Medal=='Gold')|( event.Medal=='Silver')|(event.Medal=='Bronze')][event.Sex=='F'].Team.value_counts()
tmp[(tmp.values>1)].plot('bar')
plt.xlabel('Teams')
plt.ylabel('Total medals won')
plt.title('female')
plt.figure(figsize=(20,20))
tmp=event[(event.Medal=='Gold')|( event.Medal=='Silver')|(event.Medal=='Bronze')][event.Sex=='M'].Team.value_counts()
tmp[(tmp.values>10)].plot('bar')
plt.xlabel('Teams')
plt.ylabel('Total medals won')
plt.title('male')
sport=event[(event.Medal=='Gold')|( event.Medal=='Silver')|(event.Medal=='Bronze')].set_index('Name').Sport.reset_index()
sport=sport.drop_duplicates(keep='first')
#sport.sort_values('Name')
Participants=pd.DataFrame(columns=['Gold','Silver','Bronze','Total'])
#Participants.append(event[event.Medal=='Gold'].Name.value_counts())
Participants['Gold']=event[event.Medal=='Gold'].Name.value_counts()
Participants['Silver']=event[event.Medal=='Silver'].Name.value_counts()
Participants['Bronze']=event[event.Medal=='Bronze'].Name.value_counts()
Participants['Total']=event[(event.Medal=='Gold')|( event.Medal=='Silver')|(event.Medal=='Bronze')].Name.value_counts()

Participants=pd.merge(Participants,sport,left_index=True,right_on='Name')
Participants.sort_values('Total',ascending=False,inplace=True)
Participants.set_index('Name')
x=event[((event.Medal=='Gold')|( event.Medal=='Silver')|(event.Medal=='Bronze'))]
x=x.groupby(['Team','Sport','Medal'])['Medal'].count()
#x.to_frame()#.sort_values('Medal',ascending=False)
x.to_frame()



