from IPython.display import Image
Image(filename='../input/olypic-symbol/download.jpg')

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import  seaborn  as   sns
df = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
df.head()
df[['Age','Height','Weight']].describe()
sns.barplot(x='Sex',y='Age',data=df)
sns.countplot(x='Sex',data=df)
plt.title('Male VS Female participation in Olympic',size=10,color='red')
g = sns.FacetGrid(data=df,col='Sex')
g.map(plt.hist,'Age')
f,ax=plt.subplots(figsize=(10,8))
sns.kdeplot(df['Weight'].dropna(),color='r',label='Weight')
sns.kdeplot(df['Height'].dropna(),color='b',label='Height')
g = sns.FacetGrid(df, col="Season",row ="Medal",hue='Sex')

g = g.map(plt.scatter, "Year", "Age").add_legend()
WOLRD =df.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

B =WOLRD.plot.bar(figsize=(15,4))
B.set_xticklabels(labels=df['Games'],rotation=90)

ax.set_xlabel('Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Athletes in each type of  Olympic game', size=15, color="BLUE")
Summer=df[df.Season.notnull()]
Summer_olympics=Summer[Summer.Season=='Summer']
Summer_olympics.head()
ASummer =Summer_olympics.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

C =ASummer.plot.bar(figsize=(15,4))
B.set_xticklabels(labels=Summer_olympics['Games'],rotation=90)

ax.set_xlabel('Summer Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Athletes in Summer  Olympic game', size=15, color="BLUE")
# Top 3 Countries  in Olympics All time
plt.subplot(3,1,1)
Gold_Medal  = df[df.Medal ==  "Gold"].Team.value_counts().head(3)
Gold_Medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Gold Medals")
plt.subplot(3,1,2)
silver_medal = df[df.Medal == "Silver"].Team.value_counts().head(3)
silver_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Silver Medals")
plt.subplot(3,1,3)
bronze_medal = df[df.Medal == "Bronze"].Team.value_counts().head(3)
bronze_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Bronze Medals")
plt.subplot(3,1,1)
Gold_Medal  = Summer_olympics[Summer_olympics.Medal ==  "Gold"].Team.value_counts().head(3)
Gold_Medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Gold Medals")
plt.subplot(3,1,2)
silver_medal = Summer_olympics[Summer_olympics.Medal == "Silver"].Team.value_counts().head(3)
silver_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Silver Medals")
plt.subplot(3,1,3)
bronze_medal = Summer_olympics[Summer_olympics.Medal == "Bronze"].Team.value_counts().head(3)
bronze_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Bronze Medals")

USA_Summer=Summer_olympics[Summer_olympics.Medal.notnull()]
USA_Summer_Olympics=USA_Summer[USA_Summer.Team=='United States']
USA_Summer_Olympics.head()
USASummer =USA_Summer_Olympics.groupby('Games').count()['Medal']
fig, ax = plt.subplots(figsize=(15,5))

D=USASummer.plot.bar(figsize=(15,4))
D.set_xticklabels(labels=USA_Summer_Olympics['Games'],rotation=90)

ax.set_xlabel('Summer Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Medals won', size=10, color="GREEN")
ax.set_title(' Performance of USA  in Summer  Olympic game', size=15, color="BLUE")
gold2 = USA_Summer_Olympics[(USA_Summer_Olympics.Medal == 'Gold')]

gold2.Event.value_counts().reset_index(name='Medal').head(5)
USA_Summer2=Summer_olympics[Summer_olympics.Team.notnull()]
USA_Summer_ololympics2=USA_Summer2[USA_Summer2.Team=='United States']
print('The youngest age athlete  of  USA in Summer Olympics is:' ,USA_Summer_ololympics2.Age.min())
print('The average age of athletes of USA in Summer Olympics is:',USA_Summer_ololympics2.Age.mean())
print('The oldest  Age of athlete of USA in Summer Olympics is:',USA_Summer_ololympics2.Age.max())

Bsl=USA_Summer_Olympics.pivot_table(values='Age',index='Medal',columns='Year')
f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(Bsl,annot=True, linewidths=0.05,cmap="coolwarm")
ax.set_xlabel('Summer Game Year', size=14, color="Purple")
ax.set_ylabel('Medal', size=14, color="purple")
ax.set_title(' Avg Age of USA  Athelete won Medal in Summer  Olympic games', size=18, color="Purple")
USASummer2 =USA_Summer_ololympics2.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

D=USASummer2.plot.bar(figsize=(15,4))
D.set_xticklabels(labels=USA_Summer_ololympics2['Games'],rotation=90)

ax.set_xlabel('Summer Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Participation of USA  in Summer  Olympic game', size=15, color="BLUE")
Winter=df[df.Season.notnull()]
Winter_olympics=Winter[Winter.Season=='Winter']
Winter_olympics.head()
AWinter =Winter_olympics.groupby('Games').count()['ID']
fig, ax = plt.subplots(figsize=(15,5))

C =AWinter.plot.bar(figsize=(15,4))
B.set_xticklabels(labels=Winter_olympics['Games'],rotation=90)

ax.set_xlabel('Winter Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Athletes Participated', size=10, color="GREEN")
ax.set_title(' Athletes in Winter  Olympic game', size=15, color="BLUE")
print('The youngest Age of athlete  of In winter Olympic  is:' ,Winter_olympics.Age.min())
print('The average age of athletes of Winter  Olympic id:',Winter_olympics.Age.mean())
print('The Age of  oldest athlete of Winter Olypic is',Winter_olympics.Age.max())
plt.subplot(3,1,1)
plt.title('Best Performer in Winter Olympics',size=18, color="Purple")
Gold_Medal  = Winter_olympics[Winter_olympics.Medal ==  "Gold"].Team.value_counts().head(5)
Gold_Medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Gold Medals")
plt.subplot(3,1,2)
silver_medal = Winter_olympics[Winter_olympics.Medal == "Silver"].Team.value_counts().head(5)
silver_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Silver Medals")
plt.subplot(3,1,3)
bronze_medal = Winter_olympics[Winter_olympics.Medal == "Bronze"].Team.value_counts().head(5)
bronze_medal.plot(kind='bar',rot=0,figsize=(15, 10))
plt.ylabel("Bronze Medals")


USA_Winter=Winter_olympics[Winter_olympics.Medal.notnull()]
USA_Winter_Olympics=USA_Winter[USA_Winter.Team=='United States']
USA_Winter_Olympics.head()
USAWinter =USA_Winter_Olympics.groupby('Games').count()['Medal']
fig, ax = plt.subplots(figsize=(15,5))

E=USAWinter.plot.bar(figsize=(15,4))
E.set_xticklabels(labels=USA_Winter_Olympics['Games'],rotation=90)

ax.set_xlabel('Winter Olympic Games', size=14, color="GREEN")
ax.set_ylabel('Number of Medal won', size=10, color="GREEN")
ax.set_title(' Performance of USA  in Winter Olympic game', size=15, color="BLUE")
USA_Winter_Olympics.head()
sns.countplot(x='Medal',data=USA_Winter_Olympics)
USA_Winter2=Winter_olympics[Winter_olympics.Team.notnull()]
USA_Winter_Olympics2=USA_Winter2[USA_Winter2.Team=='United States']
print('The youngest age of athlete  of  USA in Winter Olympics is:' ,USA_Winter_Olympics2.Age.min())
print('The average age of athletes of USA Winter Olympics is:',USA_Winter_Olympics2.Age.mean())
print('The oldest Age of athlete of USA Winter Olympics is :',USA_Winter_Olympics2.Age.max())
asl=USA_Winter_Olympics.pivot_table(values='Age',index='Medal',columns='Year')
f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(asl,annot=True, linewidths=0.05,cmap="coolwarm")
ax.set_xlabel('Winter Game Year', size=14, color="Purple")
ax.set_ylabel('Medal', size=14, color="purple")
ax.set_title(' Avg Age of USA  Athelete won Medal in Winter  Olympic games', size=18, color="Purple")
gold1 = USA_Winter_Olympics2[(USA_Winter_Olympics2.Medal == 'Gold')]

gold1.Event.value_counts().reset_index(name='Medal').head(5)
