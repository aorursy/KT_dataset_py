import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
import warnings
warnings.filterwarnings('ignore')
# Extract the testing and train data into a dataframe
# There was a data leak in the orginal data set, so a new data set was realeased (V2)
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
len(set(train['Id']))

# Lets take a quick look at the data
train.head()
train.isna().sum()
# We have 1 Nan in our target variable
train[train['winPlacePerc'].isna() == True]

# Lets have a look at all the players in this match
matchId = '224a123c53e008'
data = train[train['matchId'] == matchId]
data
# Remove the data where the target variable is NULL; include a sanity check
print ('length of the training set before: ',len(train))
train = train[train['winPlacePerc'].isna() != True]
print ('length of the training set after: ',len(train))

# Summary statistics for the number of kills
print('The average person kills {:.4f} players'.format(train['kills'].mean()))
print('50% of people have ',train['kills'].quantile(0.50),' kills or less')
print('75% of people have ',train['kills'].quantile(0.75),' kills or less')
print('99% of people have ',train['kills'].quantile(0.99),' kills or less')
print('while the most kills recorded in the data is', train['kills'].max())

data = train.copy()
data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize=(20,15))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title('Kill Count',fontsize=15)
plt.show()
# Summary statistics for the number of kills
print('The average person kills {:.4f} players in a short time'.format(train['killStreaks'].mean()))
print('50% of people have ',train['killStreaks'].quantile(0.50),' kills or less in a short time')
print('75% of people have ',train['killStreaks'].quantile(0.75),' kills or less in a short time')
print('99% of people have ',train['killStreaks'].quantile(0.99),' kills or less in a short time')
print('While the most kills in a row recorded in the data is', train['killStreaks'].max())

sns.jointplot(x='winPlacePerc', y='killStreaks', data=train, ratio=3, color='r')
plt.show()
# Summary statistics for the number of kills
print('The average person kills {:.4f} players from a vehicle'.format(train['roadKills'].mean()))
print('50% of people have ',train['roadKills'].quantile(0.50),' kills or less from a vehicle')
print('75% of people have ',train['roadKills'].quantile(0.75),' kills or less from a vehicle')
print('99% of people have ',train['roadKills'].quantile(0.99),' kills or less from a vehicle')
print('While the most kills recorded from a vehicle in the data is', train['roadKills'].max())

sns.jointplot(x='winPlacePerc', y='roadKills', data=train, ratio=3, color='r')
plt.show()
# Summary statistics for the number of kills
print('The average person kills {:.4f} players on their own team'.format(train['teamKills'].mean()))
print('50% of people have killed ',train['teamKills'].quantile(0.50),' team players')
print('75% of people have killed ',train['teamKills'].quantile(0.75),' team players')
print('99% of people have killed ',train['teamKills'].quantile(0.99),' team players')
print('while the most kills recorded in the data is', train['teamKills'].max())

sns.jointplot(x='winPlacePerc', y='teamKills', data=train, ratio=3, color='r')
plt.show()
# Summary statistics for the number of kills
print('The average person make {:.4f} head shots'.format(train['headshotKills'].mean()))
print('75% of people make {:.2f} head shots ',format(train['headshotKills'].quantile(0.75)))
print('99% of people make {:.2f} head shots ',format(train['headshotKills'].quantile(0.99)))
print('while the most head shots recorded in the data is', train['headshotKills'].max())

train[train['headshotKills'] == train['headshotKills'].max()]
data = train.copy()

# Keep only those players that didn't kill anyone
data = data[data['kills']==0]
plt.figure(figsize=(15,10))
plt.title('Damage Dealt by 0 killers',fontsize=15)
sns.distplot(data['damageDealt'])
plt.show()
sns.jointplot(x='winPlacePerc', y='damageDealt', data=train, ratio=3, color='r')
plt.show()
# Summary statistics for the distance walked.
print('The average person walks/runs {:.2f} m'.format(train['walkDistance'].mean()))
print('25% of people have walked/ran {:.2f} m or less'.format(train['walkDistance'].quantile(0.25)))
print('50% of people have walked/ran {:.2f} m or less'.format(train['walkDistance'].quantile(0.50)))
print('75% of people have walked/ran {:.2f} m or less'.format(train['walkDistance'].quantile(0.75)))
print('99% of people have walked/ran {:.2f} m or less'.format(train['walkDistance'].quantile(0.99)))
print('The longest distance travelled by feet in the data is {:.2f} m'.format(train['walkDistance'].max()))

sns.jointplot(x='winPlacePerc', y='walkDistance', data=train, ratio=3, color='r')
plt.show()
# Summary statistics for the distance drove
print('The average person drove {:.2f} m'.format(train['rideDistance'].mean()))
print('50% of people have drove {:.2f} m or less'.format(train['rideDistance'].quantile(0.50)))
print('75% of people have drove {:.2f} m or less'.format(train['rideDistance'].quantile(0.75)))
print('99% of people have drove {:.2f} m or less'.format(train['rideDistance'].quantile(0.99)))
print('The longest distance travelled by vehicle in the data is {:.2f} m'.format(train['rideDistance'].max()))

sns.jointplot(x='winPlacePerc', y='rideDistance', data=train, ratio=3, color='r')
plt.show()
# Summary statistics for the distance drove
print('The average person drove {:.2f} m'.format(train['swimDistance'].mean()))
print('75% of people have drove {:.2f} m or less'.format(train['swimDistance'].quantile(0.75)))
print('99% of people have drove {:.2f} m or less'.format(train['swimDistance'].quantile(0.99)))
print('The longest distance travelled by vehicle in the data is {:.2f} m'.format(train['swimDistance'].max()))

sns.jointplot(x='winPlacePerc', y='swimDistance', data=train, ratio=3, color='r')
plt.show()
# Create a new feature for total distance travelled
data = train[['winPlacePerc']].copy()
data['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']

# Summary statistics for the total distance travelled
print('The average person travelled {:.2f} m'.format(data['totalDistance'].mean()))
print('25% of people have travelled {:.2f} m or less'.format(data['totalDistance'].quantile(0.25)))
print('50% of people have travelled {:.2f} m or less'.format(data['totalDistance'].quantile(0.50)))
print('75% of people have travelled {:.2f} m or less'.format(data['totalDistance'].quantile(0.75)))
print('99% of people have travelled {:.2f} m or less'.format(data['totalDistance'].quantile(0.99)))
print('The longest distance travelled in the data is {:.2f} m'.format(data['totalDistance'].max()))

sns.jointplot(x='winPlacePerc', y='totalDistance', data=data, ratio=3, color='r')
plt.show()
# Summary statistics for the number of healing items used
print('The average person uses {:.2f} healing items'.format(train['heals'].mean()))
print('50% of people used {:.2f} healing items'.format(train['heals'].quantile(0.50)))
print('75% of people used {:.2f} healing items or less'.format(train['heals'].quantile(0.75)))
print('99% of people used {:.2f} healing items or less'.format(train['heals'].quantile(0.99)))
print('The doctor of the data used {:.2f} healing items'.format(train['heals'].max()))
# Descriptive statistics for the number of boosting items used
print('The average person uses {:.2f} boosting items'.format(train['boosts'].mean()))
print('50% of people used {:.2f} boosting items'.format(train['boosts'].quantile(0.50)))
print('75% of people used {:.2f} boosting items or less'.format(train['boosts'].quantile(0.75)))
print('99% of people used {:.2f} boosting items or less'.format(train['boosts'].quantile(0.99)))
print('The addict of the data used {:.2f} boosting items'.format(train['boosts'].max()))
data = train.copy()
data = data[data['heals'] < data['heals'].quantile(0.99)]
data = data[data['boosts'] < data['boosts'].quantile(0.99)]

f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='heals',y='winPlacePerc',data=data,color='red',alpha=1.0)
sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='blue',alpha=0.8)
plt.text(4,0.6,'Heals',color='red',fontsize = 17,style = 'italic')
plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')
plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Heals vs Boosts',fontsize = 20,color='blue')
plt.grid()

plt.show()
sns.jointplot(x='winPlacePerc', y='heals', data=train, ratio=3, color='r')
plt.show()
sns.jointplot(x='winPlacePerc', y='boosts', data=train, ratio=3, color='r')
plt.show()
data = train.copy()
# Summary statistics for the total rankPoints
print('The average person lasts {:.2f} seconds'.format(data['matchDuration'].mean()))
print('The shortest game lasted {:.1f} seconds'.format(data['matchDuration'].min()))
print('25% of people last {:.1f} seconds or less'.format(data['matchDuration'].quantile(0.25)))
print('50% of people have {:.1f} seconds or less'.format(data['matchDuration'].quantile(0.50)))
print('75% of people have {:.1f} seconds or less'.format(data['matchDuration'].quantile(0.75)))
print('99% of people have {:.1f} secondsor less'.format(data['matchDuration'].quantile(0.99)))
print('The longest game lasted {:.1f} seconds'.format(data['matchDuration'].max()))

sns.jointplot(x='winPlacePerc', y='matchDuration', data=data, ratio=3, color='r')
plt.show()
data.corr()['winPlacePerc']['matchDuration']
# Keep only the players that won the match
data = train[train['winPlacePerc'] == 1]

plt.figure(figsize=(15,10))
plt.title('Match duration for winners',fontsize=15)
sns.distplot(data['matchDuration'])
plt.show()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
# Lets look at only those matches with more than 50 players.
data = train[train['playersJoined'] > 50]

plt.figure(figsize=(15,15))
sns.countplot(data['playersJoined'].sort_values())
plt.title('Number of players joined each match',fontsize=15)
plt.show()
modes = set(train['matchType'])
print (modes)

plt.figure(figsize=(20,10))
sns.countplot(train['matchType'].sort_values())
plt.title('Match Mode Count',fontsize=15)
plt.show()
def standard_matchType(data):
    data['matchType'][data['matchType'] == 'normal-solo'] = 'solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'duo'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Other'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Other'

    return data


data = train.copy()
data = standard_matchType(data)
print (set(data['matchType']))
plt.figure(figsize=(15,10))
sns.countplot(data['matchType'].sort_values())
plt.title('Match Mode Count',fontsize=15)
plt.show()
solo = data[data['matchType'] == 'solo']
duo = data[data['matchType'] == 'duo']
squad = data[data['matchType'] == 'squad']
other = data[data['matchType'] == 'Other']

print("There are {0:} ({1:.2f}%) solo games,".format(len(solo),100*len(solo)/len(data)))
print("There are {0:} ({1:.2f}%) duo games,".format(len(duo),100*len(duo)/len(data)))
print("There are {0:} ({1:.2f}%) squad games,".format(len(squad),100*len(squad)/len(data)))
print("There are {0:} ({1:.2f}%) Other games,".format(len(other),100*len(other)/len(data)))

sns.jointplot(x='winPlacePerc', y='killStreaks', data=solo, ratio=3, color='r')
plt.show()
sns.jointplot(x='winPlacePerc', y='killStreaks', data=duo, ratio=3, color='r')
plt.show()
sns.jointplot(x='winPlacePerc', y='killStreaks', data=squad, ratio=3, color='r')
plt.show()
sns.jointplot(x='winPlacePerc', y='killStreaks', data=other, ratio=3, color='r')
plt.show()
# Create a data set with the game modes seperated
data = [solo,duo,squad,other,train]
# This is just a list of all the features 
features = ['assists','boosts','damageDealt','DBNOs','headshotKills','heals',
            'killPlace','killPoints','kills','killStreaks','longestKill',
            'maxPlace','numGroups','rankPoints','revives',
            'rideDistance','roadKills','swimDistance','teamKills',
            'vehicleDestroys','walkDistance','weaponsAcquired','winPoints']
        
# Calculate the correlation matrix for each game mode, and take only the correlation of the feature with the target variable.
solo_correlation = solo.corr()['winPlacePerc']
duo_correlation = duo.corr()['winPlacePerc']
squad_correlation = squad.corr()['winPlacePerc']
other_correlation = other.corr()['winPlacePerc']
All_correlation = train.corr()['winPlacePerc']

correlation = [solo_correlation, duo_correlation, squad_correlation, other_correlation, All_correlation]


##################################################################
#  This Cell is taking too long and causing the Kernel to crash.
##################################################################

# Plot each feature for each game mode
#fig = plt.figure(figsize=(20,150))
#ax = []
#for i in range(0, len(features)):
#    for j in range (0, len(data)):
#        fig.add_subplot(len(features), len(data), (j+1)+(i*len(data)))
#        if (i == 0):
#            ax.append(sns.regplot(x='winPlacePerc', y=features[i], data=data[j]))            
#            ax[j].text(0.5, 0.95, '{:.2f}'.format(correlation[j][features[i]]), horizontalalignment='center',verticalalignment='center',transform=ax[j].transAxes)
#            
#        else:
#            ax2 = sns.regplot(x='winPlacePerc', y=features[i], data=data[j])
#            ax2.text(0.5, 0.95, '{:.2f}'.format(correlation[j][features[i]]), horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
#
#
#ax[0].set_title('Solo')
#ax[1].set_title('Duo')
#ax[2].set_title('Squad')
#ax[3].set_title('Other')
#ax[4].set_title('All')
print ("\t\tsolo,\t\tduo,\t\tsquad,\t\tother,\tall")
for j in range(len(correlation[0].index)):
    print ("{0:} \t {1:.4f} \t {2:.4f} \t {3:.4f} \t {4:.4f} \t {5:.4f}".
           format(correlation[0].index[j],correlation[0][j],correlation[1][j],
                  correlation[2][j],correlation[3][j],correlation[4][j]))

f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(solo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(duo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(squad.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(other.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#other_correlation
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


