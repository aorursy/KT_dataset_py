import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet.plot import plot_plotly as go

kim = pd.read_csv("../input/late-night-talk-show-youtube-dataset/Jimmy_Kimmel.csv")
 #printing the first five rows 
kim.head()
kim.shape
kim.dropna(0,inplace=True)
kim.isnull().sum()
kim['videoCategoryLabel'].value_counts().plot(kind='barh',color='orange')
likem=kim.likeCount[kim.videoCategoryLabel == 'Music'].sum()
dislikem=kim.dislikeCount[kim.videoCategoryLabel == 'Music'].sum()
exp_vals=[likem,dislikem]
labels=['Total likes for Music Videos','Total dislikes for Music Videos']
plt.axis('equal')
explode=(0,0.5)
colors=['Yellow','Red']
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=45,labels=labels,colors=colors)
plt.show()
likem=kim.likeCount[kim.videoCategoryLabel == 'Comedy'].mean()
dislikem=kim.dislikeCount[kim.videoCategoryLabel == 'Entertainment'].mean()
exp_vals=[likem,dislikem]
labels=['Mean likes for Comedy Videos','Mean Likes for Entertainment Videos']
plt.axis('equal')
explode=(0,0.75)
colors=['Pink','Purple']
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=360,labels=labels,colors=colors)
plt.show()
kim.drop(['licensedContent'],axis='columns',inplace=True)
sns.heatmap(kim.corr(), annot=True)
plt.title('Corelation Matrix');
kim['publishedAtSQL']=pd.to_datetime(kim['publishedAtSQL'])
kim['publishedAtSQL'].head()
timeplot=kim[['publishedAtSQL','viewCount']].copy()
timeplot['just_date'] = timeplot['publishedAtSQL'].dt.date
timeplot=timeplot.drop('publishedAtSQL',axis=1)

timeplot.set_index('just_date', inplace= True)

timeplot.head()
plt.figure(figsize=(12,6))
plt.plot(timeplot,color='Black')
plt.title('Time Series')
plt.xlabel('Year')
plt.ylabel('Views')
plt.show()
Np=kim.commentCount[kim.videoCategoryLabel == 'Entertainment'].sum()
Kp=kim.commentCount[kim.videoCategoryLabel== 'Comedy'].sum()
data = {'Comments For Entertainment Videos' : Np,'Comments For Comedy Videos' : Kp}
names = list(data.keys())
values = list(data.values())

fig, axs = plt.subplots(1, 2, figsize=(17, 5), sharey=True)
axs[0].bar(names, values,color='Red')
axs[1].plot(names, values,color='Orange')
fig.suptitle('Categorical Plotting')
timeplot=kim[['publishedAtSQL','dislikeCount']].copy()
timeplot['just_date'] = timeplot['publishedAtSQL'].dt.date
timeplot=timeplot.drop('publishedAtSQL',axis=1)

timeplot.set_index('just_date', inplace= True)

timeplot.head()
plt.figure(figsize=(18,7))
plt.plot(timeplot,color='Red')
plt.title('Time Series')
plt.xlabel('Year')
plt.ylabel('Dislike Count')
plt.show()

likem=kim.likeCount[kim.videoCategoryLabel == 'Entertainment'].sum()
dislikem=kim.dislikeCount[kim.videoCategoryLabel == 'Entertainment'].sum()
exp_vals=[likem,dislikem]
labels=['Total likes for Entertainment Videos','Total dislikes for Entertainment Videos']
plt.axis('equal')
explode=(0,0.5)
colors=['Blue','Pink']
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=45,labels=labels,colors=colors)
plt.show()

likem=kim.likeCount[kim.videoCategoryLabel == 'Comedy'].sum()
dislikem=kim.dislikeCount[kim.videoCategoryLabel == 'Comedy'].sum()
exp_vals=[likem,dislikem]
labels=['Total likes for Comedy Videos','Total dislikes for Comedy Videos']
plt.axis('equal')
explode=(0,0.5)
colors=['Orange','Pink']
plt.pie(exp_vals,radius=2,autopct='%0.1f%%',shadow=True,explode=explode,startangle=25,labels=labels,colors=colors)
plt.show()