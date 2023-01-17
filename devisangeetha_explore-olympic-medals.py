

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import squarify 

%matplotlib inline





import os

print(os.listdir("../input"))

sns.set()

# Any results you write to the current directory are saved as output.
events =pd.read_csv("../input/athlete_events.csv")

events.info()
events.head(n=3)
noc= pd.read_csv("../input/noc_regions.csv")

noc.info()
noc.head(n=3)
olp_data=events.merge(noc,on='NOC',how='left')

olp_data.head(n=3)
evt=olp_data.groupby(by=['Year','Season','City'], as_index=False).first()

evt=evt[['Year','Season','City']]

evt.loc[evt['Season']=='Summer'].reset_index()

evt.loc[evt['Season']=='Winter'].reset_index()
display("Number of Unique Sports in Olympics:",events['Sport'].nunique())


events.groupby('Sport',as_index=False)['Event'].count().sort_values(by='Event',ascending=False).head(10)
events.groupby(['Year','Sport'],as_index=False)['Event'].count()
plt.figure(figsize=(15, 10))

topc=olp_data.groupby('region')['Medal'].count().nlargest(10).reset_index()

sns.barplot('region','Medal',data=topc)

plt.title('Top Countries in Olympic Medals')

plt.show()
topc


c_medal=olp_data[olp_data['Season']=='Summer'].groupby(['region','Medal'])['Sex'].count().reset_index()

c_medal=c_medal.pivot('region','Medal','Sex').fillna(0).sort_values(by='Gold',ascending=False).head(20)

c_medal.plot(kind='bar')

plt.xlabel('Country')

plt.title('Medals by Country- Summer Olympics ')

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.show()
c_medal=olp_data[olp_data['Season']=='Winter'].groupby(['region','Medal'])['Sex'].count().reset_index()

c_medal=c_medal.pivot('region','Medal','Sex').fillna(0).sort_values(by='Gold',ascending=False).head(20)

c_medal.plot(kind='bar')

plt.xlabel('Country')

plt.title('Medals by Country- Winter Olympics ')

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.show()
#cols=['Sport','Height']

plt.figure(figsize=(20, 10))

#events[cols].plot(kind='box')

sns.boxplot(x='Sport',y='Height',data=events)



plt.xticks(rotation = 90)

plt.show()

smedal=olp_data[(olp_data['Season']=='Summer') & (olp_data['Medal']=='Gold')].groupby('Sport')['Sex'].count().reset_index()

plt.figure(figsize=(15, 10))

plt.stem(smedal['Sport'], smedal['Sex'] )

plt.xticks(rotation = 90)

plt.show()

plt.figure(figsize=(20, 10))

#events[cols].plot(kind='box')

sns.boxplot(x='Sport',y='Weight',data=events)



plt.xticks(rotation = 90)

plt.show()
smedal=olp_data[(olp_data['Season']=='Summer') & (olp_data['Medal']=='Silver')].groupby('Sport')['Sex'].count().reset_index()

plt.figure(figsize=(15, 10))

plt.stem(smedal['Sport'], smedal['Sex'] )

plt.xticks(rotation = 90)

plt.show()

events[events['Season']=='Summer'].Sport.nunique()
events[events['Season']=='Summer'].Sport.unique()
events[events['Season']=='Winter'].Sport.unique()
plt.figure(figsize=(15, 10))

tops=events[events['Season']=='Summer'].Sport.value_counts().head(n=10)

tops.plot(kind='bar')

plt.title('Popular Sports in Olympics - Summer')

plt.show()

plt.figure(figsize=(15, 10))

tops=events[events['Season']=='Winter'].Sport.value_counts().head(n=10)

tops.plot(kind='bar')

plt.title('Popular Sports in Olympics - Winter')

plt.show()
olp_data[olp_data['Season']=='Summer'].groupby('Year')['region'].nunique()
usa_sport=events[events['Team']=='United States']['Sport'].value_counts().head(30)

plt.figure(figsize=(20,15))

g = squarify.plot(sizes=usa_sport.values, label=usa_sport.index, 

                  value=usa_sport.values,

                  alpha=.4,color=["red","green","blue", "grey"])

g.set_title("'USA Participation in Various Sports'",fontsize=20)

g.set_axis_off()

plt.show()
ath=events[events['Season']=='Summer'].groupby('Year')['ID'].count()

plt.figure(figsize=(15, 10))

ath.plot(kind='line',color='red')

plt.xlabel('Year')

plt.ylabel('No of Athletes')

plt.show()
ath=events[events['Season']=='Winter'].groupby('Year')['ID'].count()

plt.figure(figsize=(15, 10))

ath.plot(kind='line',color='blue')

plt.xlabel('Year')

plt.ylabel('No of Athletes')

plt.show()
plt.figure(figsize=(15, 10))

topm=olp_data.groupby(['region','Year'])['Medal'].count().reset_index()

topm=topm[topm['region'].isin(olp_data['region'].value_counts()[:5].index)]

#topm=topm.pivot('region','Year','Medal').fillna(0)

sns.lineplot(topm['Year'],topm['Medal'],hue=topm['region'])
sc=events.groupby('Sex')['Medal'].count().reset_index()

plt.figure(figsize=(10, 10))

sns.set()

sns.barplot(x="Sex", y="Medal",data=sc)

plt.title('Medels by Sex')

plt.show()
plt.figure(figsize=(20, 10))

y_sex=events.groupby(['Year','Sex'])['ID'].nunique().reset_index()

y_sex=y_sex.pivot('Year','Sex','ID').fillna(0)

y_sex.plot(kind = "bar"  )

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.show()
plt.figure(figsize=(20, 10))

a_medal=events.groupby('Age')['Medal'].count().reset_index()

sns.barplot(x='Age',y='Medal',data=a_medal)

plt.title('Medals by Age')

plt.xlim(0,50)

plt.xlabel('Age')

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot('Year', 'Age', data=events,palette="hls")

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot('Sport', 'Age', data=events,palette="hls")

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize=(20, 10))

avg_Age=events[events['Season']=='Summer'].groupby('Sport')['Age'].mean().reset_index().sort_values(by='Age')

sns.barplot(avg_Age.Sport,avg_Age.Age)

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize=(20, 10))

avg_Age=events[events['Season']=='Winter'].groupby('Sport')['Age'].mean().reset_index().sort_values(by='Age')

sns.barplot(avg_Age.Sport,avg_Age.Age)

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot('Year', 'Height', data=events,palette="hls")

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot('Year', 'Weight', data=events,palette="hls")

plt.show()
plt.figure(figsize=(20, 10))

sns.jointplot("Height", "Weight", data=events, kind='reg',color='green')

plt.show()
plt.figure(figsize=(20, 10))

yr_medal=events.groupby(['Year','Sex'])['Medal'].count().reset_index()

yr_medal=yr_medal.pivot('Year','Sex','Medal')

yr_medal.plot(kind='line')

fig=plt.gcf()

#sns.lineplot(x='Year',y='Medal',hue='Sex',data=yr_medal)





plt.show()

events.groupby('Name')['Medal'].count().nlargest(5).reset_index()
events.loc[events.Medal=='Gold'].groupby('Name')['Medal'].count().nlargest(5).reset_index()
events.loc[events.Medal=='Silver'].groupby('Name')['Medal'].count().nlargest(5).reset_index()
events.loc[events.Medal=='Bronze'].groupby('Name')['Medal'].count().nlargest(5).reset_index()