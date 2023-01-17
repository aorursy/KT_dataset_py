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
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')
%%time
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train_df.head()
train_df.shape
test_df.shape
#checking for missing values
train_df.isnull().sum()
train_df['groupId'].nunique()
train_df.matchId.nunique()
plt.figure(figsize=(15,7))
sns.distplot(train_df.winPlacePerc,kde=True)
plt.title('Winning % Dist.')
print ("The average winning percentage is %0.2f "%train_df.winPlacePerc.mean())
print ("%0.2f people have not won so far."%(len(train_df[train_df.winPlacePerc==0])/train_df.shape[0]*100))
print ("%0.2f people have won every time."%(len(train_df[train_df.winPlacePerc==1])/train_df.shape[0]*100))
plt.figure(figsize=(15,7))
sns.distplot(train_df.winPoints)
plt.title('Points Distribution')
print ("Maximum points won: %d"%max(train_df.winPoints))
print ("Min points won: %d"%min(train_df.winPoints))
print ("Average points won: %0.2f"%(train_df.winPoints.mean()))
plt.figure(figsize=(15,7))
sns.distplot(train_df.walkDistance)
plt.title('Walking Distance Distribution')
print ("Maximum distance walked: %d"%max(train_df.walkDistance))
print ("Min distance walked: %d"%min(train_df.walkDistance))
print ("Average distance walked: %0.2f"%(train_df.walkDistance.mean()))
df=train_df.walkDistance[train_df.winPlacePerc==0.9].sort_values(ascending=False)
print ("Teams with a winning % of more than 0.9 walk an average of: 2137.13m")
print ("No. of such teams are: %d"%len(df))
plt.figure(figsize=(15,7))
sns.jointplot(x='walkDistance',y='winPlacePerc',data=train_df,color='green',height=15,ratio=2)
plt.show()
plt.figure(figsize=(15,7))
sns.countplot(data=train_df,x='kills',orient='h')
plt.title('Kills Count plot')
plt.figure(figsize=(15,7))
sns.jointplot(x='kills',y='winPlacePerc',data=train_df,color='red',height=10,ratio=2)
plt.show()
print ("max kills: %d"%max(train_df.kills))
plt.figure(figsize=(15,7))
sns.countplot(data=train_df,x='boosts',orient='h')
plt.title('Boosts  Used')
plt.figure(figsize=(15,7))
sns.jointplot(x='boosts',y='winPlacePerc',data=train_df,color='orange',height=10,ratio=2)
plt.show()
sum(train_df.boosts==18)
plt.figure(figsize=(15,7))
train_df.revives.value_counts().plot(kind='bar')
plt.title('Times Revived')
plt.figure(figsize=(15,7))
sns.jointplot(x='revives',y='winPlacePerc',data=train_df,color='green',height=10,ratio=2)
plt.show()
train_df.columns
plt.figure(figsize=(15,7))
sns.countplot(data=train_df,x='headshotKills',orient='h')

plt.figure(figsize=(15,7))
sns.jointplot(x='headshotKills',y='walkDistance',data=train_df,color='green',height=10,ratio=2)
plt.show()
plt.figure(figsize=(15,7))
sns.countplot(data=train_df,x='killPlace')

plt.figure(figsize=(15,7))
sns.distplot(train_df.damageDealt)
plt.title('Damage Dealt')

plt.figure(figsize=(15,7))
sns.jointplot(x='heals',y='winPlacePerc',data=train_df,color='green',height=10,ratio=2)
plt.show()
plt.figure(figsize=(15,7))
sns.violinplot(data=train_df,x='killStreaks')
plt.show()
plt.figure(figsize=(15,7))
sns.countplot(data=train_df,x='roadKills')
plt.show()
plt.figure(figsize=(15,7))
sns.distplot(train_df.rideDistance)
plt.title('Ride Distance')

plt.figure(figsize=(15,7))
sns.distplot(train_df.swimDistance)
plt.title('Swim Distance')

plt.figure(figsize=(15,7))
sns.violinplot(data=train_df,x='numGroups')
plt.show()
#train_df.numGroups
plt.figure(figsize=(15,7))
sns.pointplot(data=train_df,x='numGroups',y='winPlacePerc')
plt.figure(figsize=(15,7))
sns.countplot(data=train_df,x='numGroups')
plt.show()
#train_df.numGroups
plt.figure(figsize=(15,7))
sns.pointplot(data=train_df,x='vehicleDestroys',y='winPlacePerc')
solos = train_df[train_df['numGroups']>50]
duos = train_df[(train_df['numGroups']>25) & (train_df['numGroups']<=50)]
squads = train_df[train_df['numGroups']<=25]
print("There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(len(solos), 100*len(solos)/len(train_df), len(duos), 100*len(duos)/len(train_df), len(squads), 100*len(squads)/len(train_df),))
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='kills',y='winPlacePerc',data=solos,color='black',alpha=0.8)
sns.pointplot(x='kills',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)
sns.pointplot(x='kills',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)
plt.text(37,0.6,'Solos',color='black',fontsize = 17,style = 'italic')
plt.text(37,0.55,'Duos',color='#CC0000',fontsize = 17,style = 'italic')
plt.text(37,0.5,'Squads',color='#3399FF',fontsize = 17,style = 'italic')
plt.xlabel('Number of kills',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')
plt.grid()
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='DBNOs',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)
sns.pointplot(x='DBNOs',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)
sns.pointplot(x='assists',y='winPlacePerc',data=duos,color='#FF6666',alpha=0.8)
sns.pointplot(x='assists',y='winPlacePerc',data=squads,color='#CCE5FF',alpha=0.8)
sns.pointplot(x='revives',y='winPlacePerc',data=duos,color='#660000',alpha=0.8)
sns.pointplot(x='revives',y='winPlacePerc',data=squads,color='#000066',alpha=0.8)
plt.text(14,0.5,'Duos - Assists',color='#FF6666',fontsize = 17,style = 'italic')
plt.text(14,0.45,'Duos - DBNOs',color='#CC0000',fontsize = 17,style = 'italic')
plt.text(14,0.4,'Duos - Revives',color='#660000',fontsize = 17,style = 'italic')
plt.text(14,0.35,'Squads - Assists',color='#CCE5FF',fontsize = 17,style = 'italic')
plt.text(14,0.3,'Squads - DBNOs',color='#3399FF',fontsize = 17,style = 'italic')
plt.text(14,0.25,'Squads - Revives',color='#000066',fontsize = 17,style = 'italic')
plt.xlabel('Number of DBNOs/Assits/Revives',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Duo vs Squad DBNOs, Assists, and Revives',fontsize = 20,color='blue')
plt.grid()
plt.show()

