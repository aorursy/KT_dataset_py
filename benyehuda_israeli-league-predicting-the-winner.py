# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv('../input/israeli-premier-league-team-stats-201620/IPL_16_20.csv')
df.head()

df.info()
df.drop(['Unnamed: 0','round','teamId','teamAbr','seasonName','stage'],axis=1,inplace=True)

# df=df[['teamId','AttemptonGoal','Corner','ballPossession','YellowCard','dribble','wonAirChallenge ','accurateLongBall','accurateKeyPasses','passes','points']]
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def converter(points):

    if points==3:

        return 1

    else:

        return 0
df['win'] = df['points'].apply(converter)
sns.set_style('whitegrid')

sns.countplot(x='win',data=df,palette='coolwarm')

sns.set_style('whitegrid')

sns.countplot(x='win',hue='RedCard',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='RedCard',hue='win',data=df,palette='RdBu_r')


sns.clustermap(df[['OnTarget','win','Corner','AttemptonGoal','wonDribble','lostBall','ballPossession','RedCard','passes','steals','YellowCard','keyPasses','Assist']].corr(),annot=True)
sns.clustermap(df.corr())
sns.jointplot(x='ballPossession',y='AttemptonGoal',data=df,color='red',kind='kde');
sns.jointplot(x='ballPossession',y='longBall',data=df,color='red')
sns.set_style('whitegrid')

sns.countplot(x='YellowCard',hue='win',data=df,palette='rainbow')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['win','points','PenaltyShot_Goal','Assist','CopedGoal'],axis=1), 

                                                    df['win'], test_size=0.20, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))

dfnew=df[['win', 'AttemptonGoal', 'CopedAttemptOnTargetSavedbyGK',

        'Corner', 'Cross','RedCard',

       'ShotInsidetheArea', 'ShotOutsidetheArea', 'YellowCard',

       'accurateKeyPasses', 'accurateLongBall', 'accuratePasses',

       'airChallenge', 'attackingPasses', 'ballPossession', 'ballRecovery',

       'ballRecoveryInOppHalf', 'ballRecoveryInOwnHalf', 'blockedShot',

       'centerAttack', 'centerAttackWithShot', 'dribble', 'foul',

       'goodInterceptionGoalie', 'groundChallenge', 'keyPasses',

       'leftSideAttack', 'leftSideAttackWithShot', 'longBall', 'lostBall',

       'missedShot', 'nonAttackingPasses', 'opponentFoul', 'ownHalfLostBall',

       'passes', 'rightSideAttack', 'rightSideAttackWithShot', 'steals',

       'tackles', 'tacklesSuccess', 'wonAirChallenge', 'wonDribble',

       'wonGroundChallenge', 'dribbleconv', 'airconv', 'groundconv',

       'passconv', 'attackingofall', 'targetperc', 'tackleconv', 'keyConv',

       'longConv','Offside', 'OnTarget']]
labels=list(dfnew.columns[1:20])

labels.append('win')

dfToHeat=dfnew[labels]

plt.figure(figsize=(15,15))

sns.heatmap(dfToHeat.corr(),annot=True)
labels=list(dfnew.columns[20:40])

labels.append('win')

dfToHeat=dfnew[labels]

plt.figure(figsize=(15,15))

sns.heatmap(dfToHeat.corr(),annot=True)
dfnew.drop(['groundChallenge','leftSideAttack','ownHalfLostBall','rightSideAttack','wonDribble','tackleconv'],axis=1,inplace=True)
labels=list(dfnew.columns[40:])

labels.append('win')

dfToHeat=dfnew[labels]

plt.figure(figsize=(15,15))

sns.heatmap(dfToHeat.corr(),annot=True)
X_train, X_test, y_train, y_test = train_test_split(dfnew.drop('win',axis=1), 

                                                    dfnew['win'], test_size=0.20, 

                                                    random_state=101)
fulltrain=pd.concat([X_train, y_train], join="outer",axis=1)
sns.set_style('whitegrid')

sns.countplot(x='keyPasses',hue='win',data=fulltrain,palette='rainbow')
sns.set_style('whitegrid')

sns.countplot(x='OnTarget',hue='win',data=fulltrain,palette='rainbow')
dWinKeyPasses=fulltrain['keyPasses'][fulltrain['win']==1]

l=np.percentile(dWinKeyPasses.values,5)

l
dLoseKeyPasses=fulltrain['keyPasses'][fulltrain['win']==0]

u=np.percentile(dLoseKeyPasses.values,95)

u
toDrop = fulltrain[(fulltrain['keyPasses']>u)& (fulltrain['win']==0)]

fulltrain.drop(toDrop.index,axis=0,inplace=True)

X_train.drop(toDrop.index,axis=0,inplace=True)

y_train.drop(toDrop.index,axis=0,inplace=True)
toDrop = fulltrain[(fulltrain['keyPasses']<l)& (fulltrain['win']==1)]

fulltrain.drop(toDrop.index,axis=0,inplace=True)

X_train.drop(toDrop.index,axis=0,inplace=True)

y_train.drop(toDrop.index,axis=0,inplace=True)
toDrop = fulltrain[(fulltrain['OnTarget']>=10)& (fulltrain['win']==0)]

fulltrain.drop(toDrop.index,axis=0,inplace=True)

X_train.drop(toDrop.index,axis=0,inplace=True)

y_train.drop(toDrop.index,axis=0,inplace=True)

toDrop = fulltrain[(fulltrain['OnTarget']<=1)& (fulltrain['win']==1)]

fulltrain.drop(toDrop.index,axis=0,inplace=True)

X_train.drop(toDrop.index,axis=0,inplace=True)

y_train.drop(toDrop.index,axis=0,inplace=True)


plt.figure(figsize=(8,8))

sns.boxplot(x="win", y="targetperc", data=dfnew,palette='rainbow')
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
