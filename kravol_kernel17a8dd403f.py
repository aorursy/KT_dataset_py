from sklearn import ensemble, model_selection, metrics 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score 
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import warnings
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import wordnet, pos_tag
from nltk import WordNetLemmatizer
import numpy as np
import pandas as pd

test = pd.read_csv('../input/test_V2.csv')
train = pd.read_csv('../input/train_V2.csv')

train.head()
for col in train.columns:
    print(col, train[col].isnull().sum())
print(train[train.winPlacePerc.isnull() == True])
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
k = 5
cols = train.corr().nlargest(k, 'winPlacePerc').index
cm = train[cols].corr()
f, ax = plt.subplots(figsize=(11, 11))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
plt.show()
train['heal_boost'] = train.heals + train.boosts
sns.jointplot(x="winPlacePerc", y="heal_boost", data=train, height=10, ratio=3, color="y")
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
data = train.copy()
data = data[data['heal_boost'] < data['heal_boost'].quantile(0.99)]

sns.pointplot(x='heal_boost',y='winPlacePerc',data=data,color='#606060',alpha=0.8)
plt.xlabel('Number of hboost ',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('heal + boost/ Win Ratio',fontsize = 20,color='blue')
plt.grid()
plt.show()
#corelation hboost and target
pd.concat([train.heal_boost, train.winPlacePerc],axis = 1).corr().heal_boost[1]
train.head()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
train.head()
data = train.copy()
data = data[data['playersJoined']>49]
plt.figure(figsize=(15,10))
sns.countplot(data['playersJoined'])
plt.title("Players Joined",fontsize=15)
plt.show()
train['killsnorm'] = train['kills'] * ((100 - train['playersJoined']) / 100 + 1)
train['asistnorm'] = train['assists'] * ((100 - train['playersJoined']) / 100 + 1)
train['DBNOsnorm'] = train['DBNOs'] * ((100 - train['playersJoined']) / 100 + 1)
#train['healsnorm'] = train['heals'] * ((100 - train['playersJoined']) / 100 + 1)

train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']
train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
train['healsAndBoostsPerWalkDistance'] = train['heal_boost']/(train['walkDistance']+1) #The +1 is to avoid infinity.
train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.

train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]

train['computing'] = 1
amount_of_people_in_group = train.groupby('groupId')['groupId'].transform('count')
plt.figure(figsize=(15,10))
sns.countplot(amount_of_people_in_group)
plt.title("amount_of_people_in_group",fontsize=15)
plt.xlabel('amount')
plt.show()
train.Id.unique().shape[0] == train.shape[0]
to_add = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace',
          'killPoints', 'kills', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
          'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance','weaponsAcquired',
          'winPoints', 'heal_boost', 'killsnorm', 'asistnorm','DBNOsnorm', 'totalDistance',
          'boostsPerWalkDistance', 'healsPerWalkDistance', 'healsAndBoostsPerWalkDistance',
          'killsPerWalkDistance']

group_res_mean = train.groupby('groupId')[to_add].mean()
group_res_sum = train.groupby('groupId')[to_add].sum()
match_res = train.groupby('matchId')[to_add].mean()
train.drop('computing', axis=1, inplace=True)
train.head()
#tmp_group_mean = train.apply(lambda el: group_res_mean.loc[el.groupId],axis=1)
#tmp_group_sum = train.apply(lambda el: group_res_sum.loc[el.groupId],axis=1)
#tmp_match = train.apply(lambda el: match_res.loc[el.matchId],axis=1)
print('killsPerWalkDistance - {} \n'
      'hboostPerWalkDistance - {} \n'
      'healsPerWalkDistance - {}\n'
      'boostsPerWalkDistance - {}\n'
      'totalDistance - {}\n'
      'team - {}\n'.format(
      pd.concat([train.killsPerWalkDistance, train.winPlacePerc],axis = 1).corr().killsPerWalkDistance[1],
      pd.concat([train.healsAndBoostsPerWalkDistance, train.winPlacePerc],axis = 1).corr().healsAndBoostsPerWalkDistance[1],
      pd.concat([train.healsPerWalkDistance, train.winPlacePerc],axis = 1).corr().healsPerWalkDistance[1],
      pd.concat([train.boostsPerWalkDistance, train.winPlacePerc],axis = 1).corr().boostsPerWalkDistance[1],
      pd.concat([train.totalDistance, train.winPlacePerc],axis = 1).corr().totalDistance[1],
      pd.concat([train.team, train.winPlacePerc],axis = 1).corr().team[1])
      )
print('playersJoined - {} \n'
      'killsnorm - {} \n'
      'asistnorm - {}\n'
      'team - {}\n'.format(
      pd.concat([train.playersJoined, train.winPlacePerc],axis = 1).corr().playersJoined[1],
      pd.concat([train.killsnorm, train.winPlacePerc],axis = 1).corr().killsnorm[1],
      pd.concat([train.asistnorm, train.winPlacePerc],axis = 1).corr().asistnorm[1],
      pd.concat([train.DBNOsnorm, train.winPlacePerc],axis = 1).corr().DBNOsnorm[1])
      )
k = 5
cols = train.corr().nsmallest(k, 'winPlacePerc')
cols.winPlacePerc
train.drop(2744604, inplace=True)
-cross_val_score(ensemble.AdaBoostRegressor(base_estimator=LinearRegression(normalize=True)), train.drop(['winPlacePerc','matchType', 'Id', 'groupId', 'matchId'], axis=1), train.winPlacePerc, cv = 5, scoring='neg_mean_absolute_error').mean()
#Не реально дождаться выполнения
#cross_val_score(DecisionTreeRegressor(), train.drop(['winPlacePerc','matchType', 'Id', 'groupId', 'matchId', 'killsPerWalkDistance'], axis=1), train.winPlacePerc, cv = 5, scoring='neg_mean_absolute_error').mean()
#cross_val_score(ensemble.RandomForestRegressor(n_estimators=10, criterion='mae'), train.drop(['winPlacePerc','matchType', 'Id', 'groupId', 'matchId'], axis=1), train.winPlacePerc, cv = 5, scoring='neg_mean_absolute_error')