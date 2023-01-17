# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#setting display height,max_rows,max_columns and width to desired

pd.set_option('display.max_rows',500)

pd.set_option('display.max_columns',500)
train  = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')

print ("Train Head -->")

display(train.head())

print ("Test Head -->")

display(test.head())
print (train.shape)
print (test.shape)
train.info()
train.isna().sum()
train[train['winPlacePerc'].isnull()]
#lets drop that NaN entry

train.drop(2744604,inplace = True)
test.isna().sum()
print ("Longest Kill Recored {} Average Kill Distance {}".format(train['longestKill'].max(),train['longestKill'].mean()))

print ("Max Assists Recorded {} Average Assists {}".format(train['assists'].max(),train['assists'].mean()))

print ("Max Boost Items used {} Average Boost Items Used {}".format(train['boosts'].max(),train['boosts'].mean()))

print ("Maximum DamageDealt  {} Average Damage Dealt {}".format(train['damageDealt'].max(),train['damageDealt'].mean()))

print ("Max Boost Items used {} Average Boost Items Used {}".format(train['boosts'].max(),train['boosts'].mean()))

print ("Max Heal Items used {} Average Heal Items Used {}".format(train['heals'].max(),train['heals'].mean()))

print ("Longest Kill Streak {} Average Kill Streak Used {}".format(train['killStreaks'].max(),train['killStreaks'].mean()))

print ("Maximum Kills {} Average Kills {}".format(train['kills'].max(),train['kills'].mean()))

print ("Maximum Revives {} Average Revives {}".format(train['revives'].max(),train['revives'].mean()))

print ("Maximum Team Kills {} Average Team Kills {}".format(train['teamKills'].max(),train['teamKills'].mean()))

print ("Maximum vehicleDestroys {} Average vehicleDestroys {}".format(train['vehicleDestroys'].max(),train['vehicleDestroys'].mean()))
train['playerJoined'] = train.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize = (15,10))

sns.countplot(train[train['playerJoined']>=60]['playerJoined'])

plt.title("Players Joined")

plt.show()

plt.figure(figsize = (15,10))

sns.countplot(train[train['playerJoined']<=60]['playerJoined'])

plt.title("Players Joined")

plt.show()
# I think matches with less than 50 are not worth considering

# so gonna drop those rows

train.drop(train[train['playerJoined']<=50].index,inplace = True)

train.shape
train['playersInGroup'] = train.groupby('groupId')['groupId'].transform('count')

plt.figure(figsize=(15,10))

sns.countplot(train[train['playersInGroup']>=0]['playersInGroup'])

plt.title("Number of Players in Single Group")

plt.show()

train[train['playersInGroup']>4].shape
#Groups with players greater than 4 are not valid

# as in PUBG max size of Group is 4 so we remove them

train.drop(train[train['playersInGroup']>4].index, inplace = True)

#lets find some interesting things from data

print ('Max Kills Recored {} Average Kills person kills {} while 99% people kills {}'.format(train['kills'].max(),train['kills'].mean(),train['kills'].quantile(0.99)))

# 72 kills seems suspicious lets Plot
plt.figure(figsize=(15,10))

sns.countplot(train[train['kills']>=1]['kills'])

plt.title("Number of Kills")

plt.show()
#lets check killing with winPlacePerc

# plt.figure(figsize = (15,10))

sns.jointplot(x="winPlacePerc",y="kills",data=train,height=10,ratio=3)

plt.title("WinplacePerc vs Number of Kills")

plt.show()
#Team kills cannot be 4 or more so we have to remove this

plt.figure(figsize=(15,10))

sns.countplot(train[train['teamKills']>=4]['teamKills'])

plt.title("TeamMate Kills")

plt.show()

train[train['teamKills']>=4].shape
#removing teamKills outliers

train.drop(train[train['teamKills']>=4].index, inplace = True)
print("Max number of HeadShots by Single Person {} Average Headshots {} While 99% percent people {} ".format(train['headshotKills'].max(),train['headshotKills'].mean(),train['headshotKills'].quantile(0.99)))

### remove  outlier headshots ###
######### has to do something with MatchDuration for match duration with less than 5min to 10 minutes######

# train['check'] = train[train['matchDuration']<600]

plt.figure(figsize=(15,10))

sns.countplot(train[train['matchDuration']<600]['matchDuration'])

plt.title("Match With Duration less tha 10 Minutes")

plt.show()

train[train['matchDuration']<600].shape
#we will drop the rows with match Duration less than 10 minutes
print ("Unique id counts {} while data shape {}".format(train['Id'].nunique(),train.shape))
print("Max Number of Weapons acquired by individual {} Average Number of Weapons Acquired {} while 99% percentile {}".format(train['weaponsAcquired'].max(),train['weaponsAcquired'].mean(),train['weaponsAcquired'].quantile(0.99)))
#236 weapons acquired by an individual is of course an outlier

#lets find outliers using weapons acquired

plt.figure(figsize=(15,10))

sns.countplot(train[train['weaponsAcquired']>50]['weaponsAcquired'])

plt.show()

train[train['weaponsAcquired']>50].shape
#we will remove rows with weapons acquired more than 40 as they seems suspicious

train.drop(train[train['weaponsAcquired']>50].index, inplace = True)

#lets plot WeaponsAcquired vs winPlacePerc

# plt.figure(figsize=(15,10))

sns.jointplot(x="winPlacePerc",y="weaponsAcquired",data=train,height=10,ratio=3,color="blue")

plt.title("WinPlacePerc vs WeaponsAcquired Realtion")

plt.show()
# train.sort_values(by = ['groupId']).head()
# train['checkSolo'] = [1 if (x == "solo" or x == "solo-fpp") else 0 for x in train['matchType']]

train['assistsCheck'] = ["false" if ((x == "solo" or x == "solo-fpp") and y!=0) else "true" for x,y in zip(train['matchType'],train['assists'])]

print ("Number of assists in Solo :",train[train['assistsCheck']=="false"].shape)

train['DBNOCheck'] = ["false" if ((x == "solo" or x == "solo-fpp") and y!=0) else "true" for x,y in zip(train['matchType'],train['DBNOs'])]

print ("Number of Knocks in Solos",train[train['DBNOCheck']=="false"].shape)

#lets remove these outliers from dataset

train.drop(train[train['assistsCheck']=="false"].index, inplace = True)
# #plot winPlacePerc with Vehicle Destroyed

# plt.figure(figsize = (15,10))

# sns.countplot(train[train['vehicleDestroys']>0]['vehicleDestroys'])

# plt.title("Vehicle Destroyed")

# plt.show()

# # plt.figure(figsize = (15,10))

# sns.jointplot(x="winPlacePerc",y="vehicleDestroys",data=train,height=10,ratio=3,color="lime")

# plt.title("Vehicle Destroyed jointplot")

# plt.show()

print ("Maximum walk Distance Tracelled {} Average walk Distance Travelled {}".format(train['walkDistance'].max(),train['walkDistance'].mean()))

print ("Maximum ride Distance Tracelled {} Average ride Distance Travelled {}".format(train['rideDistance'].max(),train['rideDistance'].mean()))

print ("Maximum swim Distance Tracelled {} Average swim Distance Travelled {}".format(train['swimDistance'].max(),train['swimDistance'].mean()))

print ("Maximum Total Distance Travelled by a Person in Single Match {} Average Total Distance Travelled {}".format((train['walkDistance']+train['swimDistance']+train['rideDistance']).max(),(train['walkDistance']+train['swimDistance']+train['rideDistance']).mean()))
plt.figure(figsize = (15,10))

sns.countplot(train[train['walkDistance']>5000]['walkDistance'])

plt.title("Distance Covered by Foot")

plt.show()

plt.figure(figsize = (15,10))

sns.countplot(train[train['rideDistance']>1200]['rideDistance'])

plt.title("Distance Covered by Ride")

plt.show()
plt.figure(figsize = (15,10))

sns.countplot(train[train['swimDistance']>1200]['swimDistance'])

plt.title("Distance Covered by Swimming")

plt.show()
#lets find jointplot for WinPlacePerc vs all distances

#first Walk Distance

sns.jointplot(x="winPlacePerc",y="walkDistance",data=train,height=10,ratio=3)

plt.show()
#now check ride distance

sns.jointplot(x="winPlacePerc",y="rideDistance",data=train,height=10,ratio=3,color="black")

plt.show()
#now check swim distance

sns.jointplot(x="winPlacePerc",y="swimDistance",data=train,height=10,ratio=3,color="pink")

plt.show()
#at last total distancce

train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']

sns.jointplot(x="winPlacePerc",y="totalDistance",data=train,height=10,ratio=3,color="green")

plt.show()
df = train.copy()

df = df[(df['totalDistance']==0) & (df['weaponsAcquired']>=4)]

print ("{} peoples cheated who donot move a bit but acquired weapons ".format(df['Id'].count()))

# df.sort_values(by=['groupId']).head()

train[train['groupId']=="082950bbdd1d97"].head()
# df = train.copy()

# df = df[(df['totalDistance']==0) & (df['kills']!=0)]

# df.shape

# df.head()

df = train.copy()

df = df[(df['totalDistance']==0) & (df['kills']!=0)]

print ("{} peoples cheated who donot move a bit but acquired weapons ".format(df['Id'].count()))

df.sort_values(by=['groupId']).head()
#lets see an entry and observe

train[train['groupId']=="0000a9f58703c5"].head()
plt.figure(figsize=(15,10))

sns.countplot(train[train['numGroups']>1]['numGroups'])

plt.title("Number of Groups")

plt.show()
# train['matchType'] = [1 if match =="solos" 2 elif match =="duos" else 3 for match in train['matchType']]

# df.loc[df.set_of_numbers <= 4, 'equal_or_lower_than_4?'] = 'True' 

# df.loc[df.set_of_numbers > 4, 'equal_or_lower_than_4?'] = 'False' 

train.loc[train.matchType == "solo",'matchType'] = 1

train.loc[train.matchType == "duo",'matchType'] = 2

train.loc[train.matchType == "squad",'matchType'] = 3

train.loc[train.matchType == "solo-fpp",'matchType'] = 4

train.loc[train.matchType == "duo-fpp",'matchType'] = 5

train.loc[train.matchType == "squad-fpp",'matchType'] = 6

train.loc[(train.matchType != "solo") & (train.matchType != "duo") & (train.matchType != "squad") & (train.matchType != "solo-fpp") & (train.matchType != "duo-fpp") & (train.matchType != "squad-fpp"),'matchType'] = 7

train.head()