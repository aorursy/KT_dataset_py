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
train_data=pd.read_csv('../input/train_V2.csv')
train_data.info()
train_data.head()
test_data=pd.read_csv('../input/test_V2.csv')
test_data.head()
print("Shape of Train data : "+str(train_data.shape))

print("Shape of Test data : "+str(test_data.shape))
train_data.columns
#CHECKING MISSING VALUES

train_data.isnull().sum()
#Find missing value in winPlacePerc

train_data[train_data.winPlacePerc.isnull()]
#Eliminating 2744604 from train_data

train_data.drop(2744604,inplace=True)
#CHECKING MISSING VALUES AGAIN

train_data.isnull().sum()

#MATCHES PLAYED IN TRAIN_DATA AND TEST_DATA.

#cHECKING UNIQUE MATCH-ID

len(train_data.matchId.unique())
len(test_data.matchId.unique())
#Find players in matches

train_data.groupby('matchId')['matchId'].count()
#NEW FEATURE PLAYERS PLAYED

train_data['playersPlayed']=train_data.groupby('matchId')['matchId'].transform('count')

print(train_data['playersPlayed'])
import matplotlib.pyplot as plt

import seaborn as sns

plt.subplots(figsize=(15,8))

sns.countplot(train_data[train_data['playersPlayed']>=60]['playersPlayed'])

#FINDING THE CHEATER | OUTLIERS
#TOTAL DISTANCE 

#TRAIN_DATA

train_data['totalDistance']=train_data.rideDistance+train_data.walkDistance+train_data.swimDistance
train_data['totalDistance'].head(10)
#TOTAL DISTANCE 

#TEST_DATA

test_data['totalDistance']=test_data.rideDistance+test_data.walkDistance+test_data.swimDistance
train_data['totalDistance'].head(10)
#Kills without moving

#train

train_data['killsWithoutMove']=((train_data['kills']>0)&(train_data['totalDistance']==0))

#test

test_data['killsWithoutMove']=((test_data['kills']>0)&(test_data['totalDistance']==0))
train_data['killsWithoutMove'].value_counts()

test_data['killsWithoutMove'].value_counts()
train_data.drop(train_data[train_data['killsWithoutMove']==True].index,inplace=True)

#test_data.drop(test_data[test_data['killsWithoutMove']==True].index,inplace=True)
test_data['killsWithoutMove'].value_counts()
train_data['killsWithoutMove'].value_counts()
train_data['roadKills'].value_counts()
test_data['roadKills'].value_counts()
((train_data['roadKills']>=10)&(train_data['heals']==0)&(train_data['boosts']==0)).value_counts()
((test_data['roadKills']>=10)&(test_data['heals']==0)&(test_data['boosts']==0)).value_counts()
train_data.drop(train_data[((train_data['roadKills']>=10)&(train_data['heals']==0)&(train_data['boosts']==0))==True].index,inplace=True)
test_data.drop(test_data[((test_data['roadKills']>=10)&(test_data['heals']==0)&(test_data['boosts']==0))==True].index,inplace=True)
((test_data['roadKills']>=10)&(test_data['heals']==0)&(test_data['boosts']==0)).value_counts()
# kills count



train_data['kills'].value_counts()
#test

test_data['kills'].value_counts()
#plot the kills graph

plt.subplots(figsize=(15,8))

sns.countplot(data=train_data,x=train_data['kills'])

plt.title('Kills')

plt.show()
((train_data['kills']>=25)&(train_data['heals']==0)&(train_data['boosts']==0)).value_counts()
(train_data['kills']>=35).value_counts()
# drop the outliers.



#train

train_data.drop(train_data[train_data['kills']>=35].index,inplace=True)



#test



#test_data.drop(test_data[test_data['kills']>=35].index,inplace=True)
(train_data['kills']>=35).value_counts()
(test_data['kills']>=35).value_counts()
((train_data['kills']>=25)&(train_data['heals']==0)&(train_data['boosts']==0)).value_counts()
# drop the outliers.



#train

train_data.drop(train_data[((train_data['kills']>=25)&(train_data['heals']==0)&(train_data['boosts']==0))==True].index,inplace=True)



#test



#test_data.drop(test_data[((test_data['kills']>=25)&(test_data['heals']==0)&(test_data['boosts']==0))==True].index,inplace=True)
((train_data['kills']>=25)&(train_data['heals']==0)&(train_data['boosts']==0)).value_counts()
#Longest kill

plt.subplots(figsize=(15,8))

sns.distplot(train_data['longestKill'],bins=20)
(train_data['longestKill']>=1000).value_counts()
#longest kills outliers 

#train

train_data.drop(train_data[train_data['longestKill']>=1000].index,inplace=True)



#test

#test_data.drop(test_data[test_data['longestKill']>=1000].index,inplace=True)
#CHECKING OUTLIERS

#TRAIN

(train_data['longestKill']>=1000).value_counts()
#CHECKING OUTLIERS

#TEST

(test_data['longestKill']>=1000).value_counts()
train_data[['rideDistance','walkDistance','swimDistance','totalDistance']].head()
test_data[['rideDistance','walkDistance','swimDistance','totalDistance']].head()
# plot walkDistance 



plt.figure(figsize=(15,8))

sns.distplot(train_data['walkDistance'], bins=10)

plt.show()
#Remove the outliers



#train

train_data.drop(train_data[train_data['walkDistance']>=10000].index,inplace=True)



#test

#test_data.drop(test_data[test_data['walkDistance']>=10000].index,inplace=True)
# plot rideDistance



plt.subplots(figsize=(12,4))

sns.distplot(train_data.rideDistance,bins=10)

plt.show()
#Remove the outliers.

#test

train_data.drop(train_data[train_data.rideDistance >=15000].index, inplace=True)



#test

#test_data.drop(test_data[test_data.rideDistance >=15000].index, inplace=True)
(train_data['rideDistance']>=15000).value_counts()
(test_data['rideDistance']>=15000).value_counts()
#SWIMDISTANCE

plt.subplots(figsize=(12, 4))

sns.distplot(train_data.swimDistance,bins=10)
(train_data['swimDistance']>=1000).value_counts()
#Remove the ouliers.

#SwimDistance



#train

train_data.drop(train_data[train_data.swimDistance>=1000].index,inplace=True)



#test

#test_data.drop(test_data[test_data.swimDistance>=1000].index,inplace=True)
(train_data['swimDistance']>=1000).value_counts()
(test_data['swimDistance']>=1000).value_counts()
(train_data['totalDistance']>=15000).value_counts()
#Remove the outliers.



#train

train_data.drop(train_data[train_data.totalDistance>=15000].index,inplace=True)



#test

#test_data.drop(test_data[test_data.totalDistance>=15000].index,inplace=True)
(train_data['totalDistance']>=15000).value_counts()
(test_data['totalDistance']>=15000).value_counts()
#REMOVING THE OUTLIERS RELATED TO HEALS
#heals 

plt.subplots(figsize=(15,8))

sns.countplot(data=train_data,x=train_data['heals'])

plt.title('Heals')

plt.show()
(train_data['heals']>=40).value_counts()
# remove the outliers.

#train

train_data.drop(train_data[train_data.heals>=40].index,inplace=True)



#test

#test_data.drop(test_data[test_data.heals>=40].index,inplace=True)
(train_data['heals']>=40).value_counts()
#weapons

plt.subplots(figsize=(15,8))

sns.countplot(data=train_data,x=train_data['weaponsAcquired'])

plt.title('weaponsAcquired')

plt.show()


#weaponsAcquired

#distplot



plt.figure(figsize=(12,4))

sns.distplot(train_data['weaponsAcquired'], bins=100)

plt.show()



(train_data['weaponsAcquired']>=50).value_counts()
# remove the outliers.



#train

train_data.drop(train_data[train_data.weaponsAcquired>=50].index,inplace=True)



#test

#test_data.drop(test_data[test_data.weaponsAcquired>=50].index,inplace=True)
(train_data['weaponsAcquired']>=50).value_counts()
train_data.shape
#Categorical Variables
# Create the dummy variable for categorical variable present in our data set.

#matchType

#train



train_data=pd.get_dummies(train_data,columns=['matchType'])



#test



test_data=pd.get_dummies(test_data,columns=['matchType'])
train_data.shape
train_data.info()
test_data.info()
#drop the all unnecessary columns

#train_data

train_data.drop(['killsWithoutMove'],axis=1,inplace=True)



#drop the all unnecessary columns

#test_data

#test_data.drop(['killsWithoutMove'],axis=1,inplace=True)

#feature selection
# feature selectionm using algorithm itself.

from sklearn.ensemble import RandomForestRegressor
#before that first drop some unnecessary columns.



#For train

#Id #groupId  # matchId

train_data.drop(['Id','groupId','matchId'],axis=1,inplace=True)



# For test

# Save the test Id 

test_id=test_data['Id']



test_data.drop(['Id','groupId','matchId'],axis=1,inplace=True)
train_data.shape
test_data.shape
#sample data for training

# Take sample for debugging and exploration

sample = 500000

df_sample = train_data.sample(sample)
# Metric used for the PUBG competition (Mean Absolute Error (MAE))

from sklearn.metrics import mean_absolute_error
# Design model function. 



def score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
df_sample.info()
y=df_sample['winPlacePerc']

df = df_sample.drop(columns = ['winPlacePerc'])

df.shape



#train_data test split.



from sklearn.model_selection import train_test_split



X_train,X_valid,y_train,y_valid=train_test_split(df,y,test_size=0.3,random_state=40)
#model with certaion parameter. Ramdon forest algorithm with default parameters.

m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)



m1.fit(X_train, y_train)

score(m1)
# Find the important feature using random Forest algorithm.



importance=m1.feature_importances_



# Create a new Dataframe for the given feature with their importance.

data=pd.DataFrame(sorted(zip(m1.feature_importances_, df.columns)), columns=['Value','Feature'])



# Lets plot all the feature.



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=data.sort_values(by="Value", ascending=False))
# select top 25 features and re-train the model.



new_data=data.sort_values(by='Value',ascending=False)[:25]



new_data.head()




# Plot all the variables.

plt.subplots(figsize=(15,8))

sns.barplot(x='Value',y='Feature',data=new_data)



cols=new_data.Feature.values



cols
#Recreate the model using sample data.
# train and validation data

X_train,X_valid,y_train,y_valid=train_test_split(df[cols],y,test_size=0.3,random_state=40)
X_train.shape

print(X_valid.shape)
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m1.fit(X_train, y_train)

score(m1)
#Final model training 
y_final=train_data['winPlacePerc']

df_final = train_data.drop(columns = ['winPlacePerc'])

df_final.shape
X_train,X_valid,y_train,y_valid=train_test_split(df_final,y_final,test_size=0.3,random_state=40)
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m1.fit(X_train, y_train)

score(m1)
# Replace all the infnite value from our test data. In Case ?



test_data.replace([np.inf, -np.inf], np.nan)

test_data.isnull().sum()



predictions = np.clip(a = m1.predict(test_data), a_min = 0.0, a_max = 1.0)

pred_df = pd.DataFrame({'Id' : test_id, 'winPlacePerc' : predictions})



# Create submission file

pred_df.to_csv("submission.csv", index=False)
final_output=pd.read_csv('submission.csv')
final_output.head()