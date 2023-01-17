import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import xgboost

%matplotlib inline
train_df=pd.read_csv("train_lab3.csv",index_col=0)

import matplotlib.pyplot as plt

plt.scatter(train_df['shipId'], train_df['numShips'])

plt.show()

train_df=train_df.drop(["soldierId","shipId","attackId","killingStreaks","horseRideKills"],axis=1)

train_df.tail(20)

#train_df=pd.get_dummies(train_df,columns=["year"])
import pandas_profiling



pandas_profiling.ProfileReport(train_df)
y=train_df["bestSoldierPerc"]

y.head()
x=train_df.drop(["bestSoldierPerc"],axis=1)

x.tail()
x['Total distance']=x['onFootDistance']+x['horseRideDistance']+x['swimmingDistance']

# x['Teamwork']=x['assists']+x['numSaves']

x['points_per_kill']=x['killPoints']/(x['enemiesKilled']+1)

x['points_per_kill'].fillna(0, inplace=True)

x['netkill']=x['enemiesKilled']-1.2265*x['friendlyKills']

x['Damage']=50*x['knockedOutSoldiers']+100*x['throatSlits']+500*x['castleTowerDestroys']

# x['horsekillp_per_ride']=x['horseRideKills']/(x['horseRideDistance']+1)

# x['horsekillp_per_ride'].fillna(0, inplace=True)

x['kill_per_distance']=(x['netkill'])/(x['Total distance']+0.0001)

x['kill_per_distance'].fillna(0, inplace=True)

x['potionused_per_healthlost']=x['healingPotionsUsed']/(x['healthLost']+0.0001)

x['potionused_per_healthlost'].fillna(0, inplace=True)

x['respect_per_save']=x['respectEarned']/(x['numSaves']+1)

x['respect_per_save'].fillna(0, inplace=True)

x['killpoint_per_ship']=x['killPoints']/(x['numShips']+1)

x['killpoint_per_ship'].fillna(0, inplace=True)

x['fireitem_per_weapon_and_pertower']=x['greekFireItems']/(x['weaponsUsed']+1)+x['greekFireItems']/(x['castleTowerDestroys']+1)

x['fireitem_per_weapon_and_pertower'].fillna(0, inplace=True)

x.tail()
test_df=pd.read_csv("test_data.csv",index_col=0)

#test_df=pd.get_dummies(test_df,columns=["year"])

test_df.head()
SOLDIERID=pd.read_csv("test_data.csv", usecols = ['soldierId'])

SOLDIERID.head()
test_df=test_df.drop(["soldierId","shipId","attackId","killingStreaks","horseRideKills"],axis=1)

test_df['Total distance']=test_df['onFootDistance']+test_df['horseRideDistance']+test_df['swimmingDistance']

# test_df['Teamwork']=test_df['assists']+test_df['numSaves']+test_df['castleTowerDestroys']

test_df['points_per_kill']=test_df['killPoints']/(test_df['enemiesKilled']+1)

test_df['points_per_kill'].fillna(0, inplace=True)

test_df['netkill']=test_df['enemiesKilled']-1.2265*test_df['friendlyKills']

test_df['Damage']=50*test_df['knockedOutSoldiers']+100*test_df['throatSlits']+500*test_df['castleTowerDestroys']

# test_df['horsekillp_per_ride']=test_df['horseRideKills']/(test_df['horseRideDistance']+1)

# test_df['horsekillp_per_ride'].fillna(0, inplace=True)

test_df['kill_per_distance']=(test_df['netkill'])/(test_df['Total distance']+0.0001)

test_df['kill_per_distance'].fillna(0, inplace=True)

test_df['potionused_per_healthlost']=test_df['healingPotionsUsed']/(test_df['healthLost']+0.0001)

test_df['potionused_per_healthlost'].fillna(0, inplace=True)

test_df['respect_per_save']=test_df['respectEarned']/(test_df['numSaves']+1)

test_df['respect_per_save'].fillna(0, inplace=True)

test_df['killpoint_per_ship']=test_df['killPoints']/(test_df['numShips']+1)

test_df['killpoint_per_ship'].fillna(0, inplace=True)

test_df['fireitem_per_weapon_and_pertower']=test_df['greekFireItems']/(test_df['weaponsUsed']+1)+test_df['greekFireItems']/(test_df['castleTowerDestroys']+1)

test_df['fireitem_per_weapon_and_pertower'].fillna(0, inplace=True)

test_df.head(30)

#test_df=test_df.sort_values(by=['id'])
test_df.head()
pandas_profiling.ProfileReport(x)
from sklearn.datasets import load_boston

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import xgboost as xgb

import time

mx=0

for i in range(3,8,1): 

    for j in range(100,1001,100):

        for k in range(19,22,2): 

            kf = KFold(n_splits=i,shuffle=False)

            kf.get_n_splits(x)

            predicted_y = []

            expected_y = []

            start=time.time()

            for train_index, test_index in kf.split(x):

            #print("TRAIN:", train_index, "\nTEST:", test_index)

                x_train, x_test = x.loc[train_index], x.loc[test_index]

                y_train, y_test = y[train_index], y[test_index]

                model =xgboost.XGBClassifier(n_estimators=j, max_depth=k,nthread=6)

                model.fit(x_train,y_train)

                predicted_y.extend(model.predict(x_test))

                expected_y.extend(y_test)

            print(time.time() - start)    

            if accuracy_score(predicted_y,expected_y)>mx :

                mx=accuracy_score(predicted_y,expected_y)

                mxi=i

                mxj=j

                mxk=k

                print("split:{0} est:{1} depth:{2}:{3}".format(mxi,mxj,mxk,mx))

            else :

                print(".")
# kf = KFold(n_splits=4,shuffle=False)

# kf.get_n_splits(x)

# predicted_y = []

# expected_y = []

# start=time.time()

# for train_index, test_index in kf.split(x):

#             #print("TRAIN:", train_index, "\nTEST:", test_index)

#             x_train, x_test = x.loc[train_index], x.loc[test_index]

#             y_train, y_test = y[train_index], y[test_index]

#             model =xgboost.XGBClassifier(n_estimators=1000, max_depth=5,nthread=6)

#             model.fit(x_train,y_train)

#             predicted_y.extend(model.predict(x_test))

#             expected_y.extend(y_test)

# print(time.time() - start)            

# print(accuracy_score(predicted_y,expected_y))

        

             
print(confusion_matrix(predicted_y,expected_y))
feature_importance=model.feature_importances_

len(feature_importance)

#list_nouse = []

#cols_of_value = []

for i in range(x.shape[1]):

   # if i < 25 :

    #    continue

    #if feature_importance[i] < 0.02:

     #   list_nouse.append(X.columns[i])

    #else:

     #   cols_of_value.append(X.columns[i])

    print(x.columns[i],':\t',feature_importance[i])
print(predicted_y)

print(len(predicted_y))
predicted_perc=[]

predicted_perc.extend(model.predict(test_df))

print(len(predicted_perc))
SOLDIERID.head()

final  = pd.concat([SOLDIERID, pd.DataFrame(predicted_perc)], axis=1)
final.head()
final.to_csv("result_5_xgb_feature_4_1000_5.csv", index=False)