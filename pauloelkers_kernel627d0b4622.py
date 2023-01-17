# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission_V2 = pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")

test_V2 = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

train_V2 = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")
# Splitting the training data into training set and evaluation set

train_V2 = train_V2.sort_values(by=['matchId'])

np.round(train_V2.shape[0]*0.8)

tr=train_V2[:(int)(train_V2.shape[0]*0.8)-37]

te=train_V2[(int)(train_V2.shape[0]*0.8)-37:train_V2.shape[0]+1]



# only considering the standard gameTypes

tr=tr[(tr['matchType']=='solo')| (tr['matchType']=='solo-fpp')|

              (tr['matchType']=='duo')| (tr['matchType']=='duo-fpp')|

              (tr['matchType']=='squad')| (tr['matchType']=='squad-fpp')]
#Correlation Before



#cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType','PlayerInGame','killPoints','rankPoints','winPoints','vehicleDeatroys','roadKills','swimDistance']

#cols_to_fit = [col for col in tr.columns if col not in cols_to_drop]

cols_to_fit =['kills','assists','boosts','heals','damageDealt','DBNOs','headshotKills','killStreaks',

              'longestKill','matchDuration','revives','rideDistance','walkDistance','weaponsAcquired','killPlace','winPlacePerc']

corr_of = tr[cols_to_fit].corr()



plt.figure(figsize=(12,9))

sns.heatmap(

    corr_of,

    xticklabels=corr_of.columns.values,

    yticklabels=corr_of.columns.values,

    linecolor='white',

    linewidths=0.1,

    cmap="RdBu"

)

plt.show()
#New features





tr['PlayerInGame']=tr.groupby('matchId')['matchId'].transform('count')

te['PlayerInGame']=te.groupby('matchId')['matchId'].transform('count')



tr['killPlace']=tr['killPlace']/tr['PlayerInGame']

te['killPlace']=te['killPlace']/te['PlayerInGame']



tr['walkDistancet']=tr['walkDistance']/tr['matchDuration']

te['walkDistancet']=te['walkDistance']/te['matchDuration']

tr['rideDistancet']=tr['rideDistance']/tr['matchDuration']

te['rideDistancet']=te['rideDistance']/te['matchDuration']



#dealing with teamvalues



tr['groupSize']=tr.groupby('groupId')['groupId'].transform('count')

te['groupSize']=te.groupby('groupId')['groupId'].transform('count')



tr['avkills']=tr.groupby('groupId')['kills'].transform('sum')/tr['groupSize']

te['avkills']=te.groupby('groupId')['kills'].transform('sum')/te['groupSize']



tr['avassists']=tr.groupby('groupId')['assists'].transform('sum')/tr['groupSize']

te['avassists']=te.groupby('groupId')['assists'].transform('sum')/te['groupSize']



tr['avboosts']=tr.groupby('groupId')['boosts'].transform('sum')/tr['groupSize']

te['avboosts']=te.groupby('groupId')['boosts'].transform('sum')/te['groupSize']



tr['avheals']=tr.groupby('groupId')['heals'].transform('sum')/tr['groupSize']

te['avheals']=te.groupby('groupId')['heals'].transform('sum')/te['groupSize']



tr['avdamageDealt']=tr.groupby('groupId')['damageDealt'].transform('sum')/tr['groupSize']

te['avdamageDealt']=te.groupby('groupId')['damageDealt'].transform('sum')/te['groupSize']



tr['avheadshotKills']=tr.groupby('groupId')['headshotKills'].transform('sum')/tr['groupSize']

te['avheadshotKills']=te.groupby('groupId')['headshotKills'].transform('sum')/te['groupSize']



tr['avkillStreaks']=tr.groupby('groupId')['killStreaks'].transform('sum')/tr['groupSize']

te['avkillStreaks']=te.groupby('groupId')['killStreaks'].transform('sum')/te['groupSize']



tr['avrideDistance']=tr.groupby('groupId')['rideDistancet'].transform('sum')/tr['groupSize']

te['avrideDistance']=te.groupby('groupId')['rideDistancet'].transform('sum')/te['groupSize']



tr['avwalkDistance']=tr.groupby('groupId')['walkDistancet'].transform('sum')/tr['groupSize']

te['avwalkDistance']=te.groupby('groupId')['walkDistancet'].transform('sum')/te['groupSize']



tr['avweaponsAcquired']=tr.groupby('groupId')['weaponsAcquired'].transform('sum')/tr['groupSize']

te['avweaponsAcquired']=te.groupby('groupId')['weaponsAcquired'].transform('sum')/te['groupSize']



#groupexcess

tr['teammemberExcess1']=tr[(tr['matchType']=='solo')| (tr['matchType']=='solo-fpp')]['groupSize']-1

tr['teammemberExcess2']=tr[(tr['matchType']=='duo')| (tr['matchType']=='duo-fpp')]['groupSize']-2

tr['teammemberExcess4']=tr[(tr['matchType']=='squad')| (tr['matchType']=='squad-fpp')]['groupSize']-4

A=tr['teammemberExcess1']

B=tr['teammemberExcess2']

C=tr['teammemberExcess4']

I=np.isnan(A)

A[I]=B[I]

I=np.isnan(A)

A[I]=C[I]

I=np.isnan(A)

A[I]=0

I=A<=0

A[I]=0

tr['teammemberExcess']=A

tr=tr.drop(columns=['teammemberExcess1', 'teammemberExcess2','teammemberExcess4'])



te['teammemberExcess1']=te[(te['matchType']=='solo')| (te['matchType']=='solo-fpp')]['groupSize']-1

te['teammemberExcess2']=te[(te['matchType']=='duo')| (te['matchType']=='duo-fpp')]['groupSize']-2

te['teammemberExcess4']=te[(te['matchType']=='squad')| (te['matchType']=='squad-fpp')]['groupSize']-4

A=te['teammemberExcess1']

B=te['teammemberExcess2']

C=te['teammemberExcess4']

I=np.isnan(A)

A[I]=B[I]

I=np.isnan(A)

A[I]=C[I]

I=np.isnan(A)

A[I]=0

I=A<=0

A[I]=0

te['teammemberExcess']=A

te=te.drop(columns=['teammemberExcess1', 'teammemberExcess2','teammemberExcess4'])



#max killPlace of the team

tr['bestkillPlace']=tr.groupby('groupId')['killPlace'].transform('min')

te['bestkillPlace']=te.groupby('groupId')['killPlace'].transform('min')

# Correlation After

cols_to_fit =['avkills','avassists','avboosts','avheals','avdamageDealt','DBNOs','avheadshotKills','avkillStreaks',

              'longestKill','matchDuration','revives','avrideDistance','avwalkDistance','avweaponsAcquired','killPlace',

              'bestkillPlace','teammemberExcess','winPlacePerc']

corr_nf = tr[cols_to_fit].corr()



plt.figure(figsize=(12,9))

sns.heatmap(

    corr_nf,

    xticklabels=corr_nf.columns.values,

    yticklabels=corr_nf.columns.values,

    linecolor='white',

    linewidths=0.1,

    cmap="RdBu"

)

plt.show()
#Compare correlation

corr_of['winPlacePerc']
#Compare correlation

corr_nf['winPlacePerc']


finaltrain_X=tr[['avkills','avassists','avboosts','avheals','avdamageDealt','DBNOs','avheadshotKills','avkillStreaks',

              'longestKill','matchDuration','revives','avrideDistance','avwalkDistance','avweaponsAcquired','killPlace',

              'bestkillPlace','teammemberExcess']]

finaltrain_Y=tr['winPlacePerc']



#There is one NaN in finaltrain_Y we will drop it

finaltrain_Y.drop(2744604, inplace=True)

finaltrain_X.drop(2744604, inplace=True)







finaltest_X=te[['avkills','avassists','avboosts','avheals','avdamageDealt','DBNOs','avheadshotKills','avkillStreaks',

              'longestKill','matchDuration','revives','avrideDistance','avwalkDistance','avweaponsAcquired','killPlace',

              'bestkillPlace','teammemberExcess']]

finaltest_Y=te['winPlacePerc']







#from sklearn import linear_model

#regr = linear_model.LinearRegression()

#regr.fit(finaltrain_X, finaltrain_Y)
import xgboost

xregr=xgboost.XGBRegressor()

xregr.fit(finaltrain_X, finaltrain_Y)

train_pred = xregr.predict(finaltrain_X)

test_pred = xregr.predict(finaltest_X)
#train_pred = regr.predict(finaltrain_X)

#test_pred =  regr.predict(finaltest_X)
#Evaluation



for i in range(0,test_pred.size):

    if(test_pred[i]<=0):

        test_pred[i]=0

    if(test_pred[i]>=1):

        test_pred[i]=1







(np.abs(test_pred-finaltest_Y)).sum()/test_pred.size
# Making the final prediction

#New features





test_V2['PlayerInGame']=test_V2.groupby('matchId')['matchId'].transform('count')

test_V2['killPlace']=test_V2['killPlace']/test_V2['PlayerInGame']

test_V2['walkDistancet']=test_V2['walkDistance']/test_V2['matchDuration']

test_V2['rideDistancet']=test_V2['rideDistance']/test_V2['matchDuration']





#dealing with teamvalues



test_V2['groupSize']=test_V2.groupby('groupId')['groupId'].transform('count')

test_V2['avkills']=test_V2.groupby('groupId')['kills'].transform('sum')/test_V2['groupSize']

test_V2['avassists']=test_V2.groupby('groupId')['assists'].transform('sum')/test_V2['groupSize']

test_V2['avboosts']=test_V2.groupby('groupId')['boosts'].transform('sum')/test_V2['groupSize']

test_V2['avheals']=test_V2.groupby('groupId')['heals'].transform('sum')/test_V2['groupSize']

test_V2['avdamageDealt']=test_V2.groupby('groupId')['damageDealt'].transform('sum')/test_V2['groupSize']

test_V2['avheadshotKills']=test_V2.groupby('groupId')['headshotKills'].transform('sum')/test_V2['groupSize']

test_V2['avkillStreaks']=test_V2.groupby('groupId')['killStreaks'].transform('sum')/test_V2['groupSize']

test_V2['avrideDistance']=test_V2.groupby('groupId')['rideDistancet'].transform('sum')/test_V2['groupSize']

test_V2['avwalkDistance']=test_V2.groupby('groupId')['walkDistancet'].transform('sum')/test_V2['groupSize']

test_V2['avweaponsAcquired']=test_V2.groupby('groupId')['weaponsAcquired'].transform('sum')/test_V2['groupSize']



#groupexcess

test_V2['teammemberExcess1']=test_V2[(test_V2['matchType']=='solo')| (test_V2['matchType']=='solo-fpp')]['groupSize']-1

test_V2['teammemberExcess2']=test_V2[(tr['matchType']=='duo')| (test_V2['matchType']=='duo-fpp')]['groupSize']-2

test_V2['teammemberExcess4']=test_V2[(tr['matchType']=='squad')| (test_V2['matchType']=='squad-fpp')]['groupSize']-4

A=test_V2['teammemberExcess1']

B=test_V2['teammemberExcess2']

C=test_V2['teammemberExcess4']

I=np.isnan(A)

A[I]=B[I]

I=np.isnan(A)

A[I]=C[I]

I=np.isnan(A)

A[I]=0

I=A<=0

A[I]=0

test_V2['teammemberExcess']=A

test_V2=test_V2.drop(columns=['teammemberExcess1', 'teammemberExcess2','teammemberExcess4'])



#max killPlace of the team

test_V2['bestkillPlace']=test_V2.groupby('groupId')['killPlace'].transform('min')









finalfinaltest_X=test_V2[['avkills','avassists','avboosts','avheals','avdamageDealt','DBNOs','avheadshotKills','avkillStreaks',

              'longestKill','matchDuration','revives','avrideDistance','avwalkDistance','avweaponsAcquired','killPlace',

              'bestkillPlace','teammemberExcess']]



finaltest_pred = xregr.predict(finalfinaltest_X)



for i in range(0,finaltest_pred.size):

    if(finaltest_pred[i]<=0):

        finaltest_pred[i]=0

    if(finaltest_pred[i]>=1):

        finaltest_pred[i]=1

#Submitting

#finaltest_pred



submission = pd.DataFrame(

    {'Id':test_V2.Id, 'winPlacePerc':finaltest_pred},

    columns=['Id','winPlacePerc']

    )



submission.to_csv('submission.csv',index=False)