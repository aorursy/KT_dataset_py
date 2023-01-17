import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train_V2.csv') #read the training data

test = pd.read_csv('../input/test_V2.csv') #read the testing data
train.info()

test.info()
train.head() # Check the sample of the training set
test.head() # Check the smaple of testing data
print("The average person kills {:.4f} players, 99% of the people have {} kills or less, while the most kills ever recorded is {}.".format(train['kills'].mean(), train['kills'].quantile(0.99), train['kills'].max()))
data = train.copy() #refere the train dataFrame using sollow copy

data.loc[data['kills']>data['kills'].quantile(0.99)] = '8+' # set the above 99% of kill as 8+, because max kill is 7

plt.figure(figsize=(15,10))

sns.set(style = 'whitegrid')

sns.countplot(data['kills'].astype('str').sort_values())

plt.title("Kill Count", fontsize = 14)

plt.show()
data = train.copy()

data = data[data['kills'] == 0] # Now our data only referes to where kills are 0

plt.figure(figsize = (15,10))

plt.title("Damage Dealt by 0 killers", fontsize = 15)

sns.distplot(data['damageDealt']) # Distributaion over damage of people with 0 kills

plt.show()
print("{} players ({:.4f}%) have won without a single kill!".format(len(data[data['winPlacePerc'] == 1]), 100*len(data[data['winPlacePerc'] == 1])/len(train)))

data_dmg = train[train['damageDealt'] == 0].copy() # Data Frame: data_dmg has instaces where damage is 0. Whereas data has instances with kills = 0

print("{} players ({:.4f}%) have won without dealing damage!".format(len(data_dmg[data_dmg['winPlacePerc'] == 1]), 100*len(data_dmg[data_dmg['winPlacePerc'] == 1])/len(train)))
data_kd = data[data['damageDealt'] == 0] # Now, our data frame has players with 0 kills and 0 damage dealt

print("{} players ({:.4f}%) have won without a single kill and without dealing damage!".format(len(data_kd[data_kd['winPlacePerc'] == 1]), 100*(len(data_kd[data_kd['winPlacePerc'] == 1])/len(train))))
sns.jointplot(x = 'winPlacePerc', y = 'kills', data = train, height = 10, ratio = 3, color = 'r')

plt.show()
kills = train.copy()

kills['killsCategories'] = pd.cut(kills['kills'], [-1,0,3,6,10,80], labels = ['0 kills', '1-3 kills', '4-6 kills', '7-10 kills', '10+ kills']) # labeled the data respective to thier kills

plt.figure(figsize = (15,8))

sns.boxplot(x = 'killsCategories', y = 'winPlacePerc', data = kills)

plt.show()
print("The average person walks for {:.1f}m, 99% of the people have walked {}m or less, while the marathoner champion walked for {}m.".format(train['walkDistance'].mean(), train['walkDistance'].quantile(0.99), train['walkDistance'].max()))
data = train.copy()

data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]

plt.figure(figsize=(15,10))

plt.title('Walking Distance Distribution', fontsize = 15)

sns.distplot(data['walkDistance'])

plt.show()
print("{} players ({:.4f}%) walked 0 meters. This means that they die before even taking a step or they are afk(more possible)".format(len(data[data['walkDistance'] == 0]), 100*len(data_dmg[data_dmg['walkDistance'] == 0])/len(train)))
sns.jointplot(x = 'winPlacePerc', y = "walkDistance", data = train, height = 10, ratio = 3, color = 'g')

plt.show()
print("The average person drives for {:.1f}m, 99% of the people have dirved {}m or less, while the formula 1 champion drived for {}m.".format(train['rideDistance'].mean(), train['rideDistance'].quantile(0.99), train['rideDistance'].max()))
data = train.copy()

data = data[data['rideDistance'] < train['rideDistance'].quantile(0.99)]

plt.figure(figsize=(15,10))

plt.title('Ride Distance Distributaion', fontsize = 15)

sns.distplot(data['rideDistance'])

plt.show()
print("{} players ({:.4f}%) drived for 0 meters. This means that they don't have a driving licence yet.".format(len(data[data['rideDistance'] == 0]), 100*len(data_dmg[data_dmg['rideDistance']==0])/len(train)))
sns.jointplot(x="winPlacePerc", y='rideDistance',data = train, height = 10,ratio = 3, color = 'b')

plt.show()
plt.subplots(figsize=(20,10))

sns.pointplot(x='vehicleDestroys', y='winPlacePerc',data=data,color='g',alpha=0.8)

plt.xlabel("Number of Vehicles Destroyed",fontsize = 15,color='blue')

plt.ylabel("Win Percentage",fontsize = 15,color='blue')

plt.title("Vehicle Destroyed vs Win Ratio", fontsize = 20, color='blue')

plt.show()
print("The average person uses {:.1f} heal items, 99% of people use {} or less, while the doctor used {}.".format(train['heals'].mean(), train['heals'].quantile(0.99), train['heals'].max()))

print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}.".format(train['boosts'].mean(), train['boosts'].quantile(0.99), train['boosts'].max()))
data = train.copy()

data = data[data['heals'] < data['heals'].quantile(0.99)]

data = data[data['boosts'] < data['boosts'].quantile(0.99)]



plt.subplots(figsize =(20,10))

sns.pointplot(x='heals',y='winPlacePerc',data=data,color='#006400',alpha=0.8)

sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='orange',alpha=0.8)

plt.text(4,0.6,'Heals',color='lime',fontsize = 17,style = 'italic')

plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')

plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Heals vs Boosts',fontsize = 20,color='blue')

plt.show()

sns.jointplot(x="winPlacePerc", y="heals", data=train, height=10, ratio=3, color="#006400")

plt.show()
sns.jointplot(x="winPlacePerc", y="boosts", data=train, height=10, ratio=3, color="orange")

plt.show()
solos = train[train['numGroups']>50]

duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]

squads = train[train['numGroups']<=25]



print("There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games."

      .format(len(solos), 100*len(solos)/len(train), len(duos), 100*len(duos)/len(train),

              len(squads), 100*len(squads)/len(train)))
plt.subplots(figsize=(20,10))

sns.pointplot(x = 'kills', y = 'winPlacePerc', data = solos, color = 'black', alpha = 0.8)

sns.pointplot(x = 'kills', y = 'winPlacePerc', data = duos, color = 'red', alpha = 0.8)

sns.pointplot(x = 'kills', y = 'winPlacePerc', data = squads, color = 'blue', alpha = 0.8)

plt.text(37,0.6,'Solos',color='black',fontsize=17,style='italic')

plt.text(37,0.55,'Duos',color='red',fontsize=17,style='italic')

plt.text(37,0.5,'Squads',color='blue',fontsize=17,style='italic')

plt.xlabel('Number of Kills',fontsize=15,color='black')

plt.ylabel('Win Percentage',fontsize=15,color='black')

plt.title('Solo vs Duo vs Squad',fontsize=20,color='black')

plt.show()
plt.subplots(figsize =(20,10))

sns.pointplot(x='DBNOs',y='winPlacePerc',data=duos,color='red',alpha=0.8)

sns.pointplot(x='DBNOs',y='winPlacePerc',data=squads,color='blue',alpha=0.8)

sns.pointplot(x='assists',y='winPlacePerc',data=duos,color='lime',alpha=0.8)

sns.pointplot(x='assists',y='winPlacePerc',data=squads,color='green',alpha=0.8)

sns.pointplot(x='revives',y='winPlacePerc',data=duos,color='orange',alpha=0.8)

sns.pointplot(x='revives',y='winPlacePerc',data=squads,color='black',alpha=0.8)

plt.text(14,0.5,'Duos - Assists',color='lime',fontsize = 17,style = 'italic')

plt.text(14,0.45,'Duos - DBNOs',color='red',fontsize = 17,style = 'italic')

plt.text(14,0.4,'Duos - Revives',color='orange',fontsize = 17,style = 'italic')

plt.text(14,0.35,'Squads - Assists',color='green',fontsize = 17,style = 'italic')

plt.text(14,0.3,'Squads - DBNOs',color='blue',fontsize = 17,style = 'italic')

plt.text(14,0.25,'Squads - Revives',color='black',fontsize = 17,style = 'italic')

plt.xlabel('Number of DBNOs/Assits/Revives',fontsize = 15,color='black')

plt.ylabel('Win Percentage',fontsize = 15,color='black')

plt.title('Duo vs Squad DBNOs, Assists, and Revives',fontsize = 20,color='black')

plt.show()
f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()


#k = 5 #number of variables for heatmap

#f,ax = plt.subplots(figsize=(7, 7))

#cols = train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index

#cm = np.ma.corrcoef(train[cols].values.T)

#sns.set(font_scale=1.25)

#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

#                yticklabels=cols.values, xticklabels=cols.values)

#plt.show()
#sns.set()

#cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']

#sns.pairplot(train[cols], size = 2.5)

#plt.show()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
data = train.copy()

data = data[data['playersJoined']>50]

plt.figure(figsize=(15,10))

sns.countplot(data['playersJoined'])

plt.title("Players Joined",fontsize=15)

plt.show()
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']][5:8]
train['healsAndBoosts'] = train['heals']+train['boosts']

train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']
train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.

train['boostsPerWalkDistance'].fillna(0, inplace=True)

train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.

train['healsPerWalkDistance'].fillna(0, inplace=True)

train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts']/(train['walkDistance']+1) #The +1 is to avoid infinity.

train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)

train[['walkDistance', 'boosts', 'boostsPerWalkDistance' ,'heals',  'healsPerWalkDistance', 'healsAndBoosts', 'healsAndBoostsPerWalkDistance']][40:45]
train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.

train['killsPerWalkDistance'].fillna(0, inplace=True)

train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance', 'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)
train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]
train.head()
f,ax = plt.subplots(figsize=(30, 30))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
dfTrain = pd.DataFrame(train)

dfTrain.to_csv("train_V3.csv",index=False)