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
Original_Train_df = pd.read_csv('../input/train_V2.csv')

Original_Test_df = pd.read_csv('../input/test_V2.csv')

Train_df = Original_Train_df

Test_df = Original_Test_df
Train_df.head(5)

Train_df.isna().sum()
Train_df[Train_df['winPlacePerc'].isna()]

Train_df = Train_df.fillna(0)
Train_df = Train_df.drop(['Id','groupId','matchId'],axis=1)
Train_df[((Train_df['DBNOs']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))|((Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))]
Train_df = Train_df.drop(Train_df[((Train_df['DBNOs']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))|((Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0))].index,axis = 0)
Train_df[(Train_df['killPlace']==1)&(Train_df['kills']==0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0)]
Train_df = Train_df.drop(Train_df[(Train_df['killPlace']==1)&(Train_df['kills']==0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)&(Train_df['swimDistance']==0)].index,axis = 0)
Train_df[(Train_df['swimDistance']>0)&(Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)]
Train_df = Train_df.drop(Train_df[(Train_df['swimDistance']>0)&(Train_df['kills']>0)&(Train_df['walkDistance']==0)&(Train_df['rideDistance']==0)].index, axis = 0)
Train_df['headshotRate'] = Train_df['headshotKills']/Train_df['kills']

Train_df[(Train_df['kills']>10)&(Train_df['headshotRate']>=.9)]
Train_df = Train_df.drop(Train_df[(Train_df['kills']>10)&(Train_df['headshotRate']>=.9)].index,axis = 0)

Train_df[(Train_df['walkDistance']==0)&(Train_df['weaponsAcquired']>2)]
Train_df = Train_df.drop(Train_df[(Train_df['walkDistance']==0)&(Train_df['weaponsAcquired']>2)].index, axis = 0)


Train_df.kills.describe().astype(int)

Train_df[(Train_df['kills']>10)&(Train_df['walkDistance']<100)]
Train_df = Train_df.drop(Train_df[(Train_df['kills']>10)&(Train_df['walkDistance']<100)].index, axis = 0)
Train_df[(((Train_df['vehicleDestroys']>1)|(Train_df['kills']>1))&(Train_df['damageDealt']==0))]
Train_df = Train_df.drop(Train_df[(((Train_df['vehicleDestroys']>1)|(Train_df['kills']>1))&(Train_df['damageDealt']==0))].index,axis = 0)
Train_df.walkDistance.describe().astype(int)                                      

Train_df['Player_speed_m/s'] = Train_df['walkDistance']/Train_df['matchDuration']
Train_df['Player_speed_m/s'].describe().astype(int)
Train_df.corr()['Player_speed_m/s']
Train_df.corr().winPlacePerc
Train_df[((Train_df['Player_speed_m/s']>5.5)&(Train_df['walkDistance']>2000)&(Train_df['kills']>1))]
Train_df = Train_df.drop(Train_df[((Train_df['Player_speed_m/s']>5.5)&(Train_df['walkDistance']>2000)&(Train_df['kills']>1))].index, axis = 0)
Train_df.corr().winPlacePerc

Final_Train_df =Train_df.drop(['killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','teamKills','vehicleDestroys','walkDistance','winPoints','headshotRate','swimDistance'], axis = 1)
Final_Train_df.corr().winPlacePerc
Final_Train_df.info()
Final_Train_df['matchType'].nunique()
# Importing the dataset

y = Final_Train_df.loc[:, ['winPlacePerc']].values

Final_Train_df = Final_Train_df.drop(['winPlacePerc'],axis = 1)

Final_Train_df = pd.get_dummies(Final_Train_df['matchType'])

#X = Final_Train_df.loc[:, ['assists','boosts','damageDealth','DBNOs','headshotKills','heals', 'killPlace','kills','killsStreaks','matchType','longestKill','revives','rideDistane','weaponAcquired','Player_speed_m/s']].values

X = Final_Train_df.loc[:,:].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Part 2 - Now let's make the ANN!



# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

model = Sequential()



# Adding the input layer and the first hidden layer

model.add(Dense(256, activation = 'relu', input_dim = X_train.shape[1]))



# Adding the second hidden layer

model.add(Dense(units = 256, activation = 'relu'))



# Adding the third hidden layer

model.add(Dense(units = 512, activation = 'relu'))



# Adding the fourth hidden layer

model.add(Dense(units = 256, activation = 'relu'))



# Adding the fifth hidden layer

model.add(Dense(units = 256, activation = 'relu'))



# Adding the output layer

model.add(Dense(units = 1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')

model.fit(X_train, y_train, batch_size = 200, epochs = 10)
# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = model.predict(X_test)



from sklearn.metrics import mean_squared_error



mae = mean_squared_error(y_test, y_pred)

mae
from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test, y_pred))

rms