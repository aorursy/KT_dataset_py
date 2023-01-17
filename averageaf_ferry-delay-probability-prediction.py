# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from datetime import datetime

from datetime import timedelta

import plotly.express as pltly



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
traindb = pd.read_csv('../input/canssi-ncsc-ferry-delays/train.csv')
traindb.head()
traindb.shape
traindb[traindb['Delay.Indicator']==1].groupby('Status').count()['Trip']
traindb[traindb['Delay.Indicator']==1].groupby('Trip').count()['Delay.Indicator'].sum() # Count of delays for trip
traindb[traindb['Delay.Indicator']==0].groupby('Trip').count()['Delay.Indicator'].sum() # Count of trips on time
lateTrips = (traindb[traindb['Delay.Indicator']==1].groupby('Trip').sum()['Delay.Indicator'])/(traindb.groupby('Trip').count()['Delay.Indicator'])

lateTrips
lateDays = traindb[traindb['Delay.Indicator']==1].groupby('Day').count()['Delay.Indicator']/(traindb.groupby('Day').count()['Delay.Indicator'])

lateDays
lateVessels = traindb[traindb['Delay.Indicator']==1].groupby('Vessel.Name').count()['Delay.Indicator']/(traindb.groupby('Vessel.Name').count()['Delay.Indicator'])

lateVessels
traffic = pd.read_csv('../input/canssi-ncsc-ferry-delays/traffic.csv')
traffic
vaweather = pd.read_csv('../input/canssi-ncsc-ferry-delays/vancouver.csv')
vaweather.isnull().sum()
vaweather.head()
viweather = pd.read_csv('../input/canssi-ncsc-ferry-delays/victoria.csv')
viweather.isnull().sum()
traindb.head()
traffic.head()
dates = pd.to_datetime(traffic[['Year','Month','Day','Hour','Minute']])
# I don't really want seconds, since they do not appear in our train dataset.

traffic = pd.concat([traffic.drop(['Year','Day','Month','Second'],axis=1),dates],axis=1)

traffic.columns = ['Hour','Minute','Traffic.Ordinal','Date.Ordinal']
traffic
traindb['Full.Date'] = pd.to_datetime(traindb['Full.Date'] + ' ' + traindb['Scheduled.Departure'])
traindb.head()
len(traindb['Scheduled.Departure'].unique())
SchedDept = traindb['Scheduled.Departure'].unique()
traindb.drop(['Scheduled.Departure','Day.of.Month','Month','Year'],axis=1,inplace=True)
traffic.head()
SMA_8 = traffic['Traffic.Ordinal'].rolling(window=8).mean()
traffic = pd.concat([traffic,SMA_8],axis=1)

traffic.columns = ['Hour','Minute','Traffic.Ordinal','Date.Ordinal','SMA_8']
traffic[4:8]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
len(traffic)
traindb = traindb.sort_values(by=['Full.Date'])
from math import floor

def binary_search(Array, Search_Term):

    n = len(Array)

    L = 0

    R = n-1

    

    while L <= R:

        mid = floor((L+R)/2)

        if Array[mid] < Search_Term:

            L = mid + 1

        elif Array[mid] > Search_Term:

            R = mid - 1

        else:

            return mid

    return -1

SMA8col = pd.DataFrame(columns=['SMA_8'])

for i in range(0,len(traindb)):

    dateWanted = traindb['Full.Date'][i]

    k=0

    minOffset = [1,-2,2,-3,3]

    # Consider that traindb has departures before capture of traffic data began

    # We assign a 1 in traffic sma for this time period

    if dateWanted < traffic['Date.Ordinal'][0]:  # If the datetime is before data capture beginning date

        SMA8col = SMA8col.append({'SMA_8': 1},ignore_index=True)  

    elif (dateWanted.year == 2017 and dateWanted.month == 7 and dateWanted.day == 7) or (dateWanted.year == 2017 and dateWanted.month == 3 and dateWanted.day == 10):

        SMA8col = SMA8col.append({'SMA_8': 0},ignore_index=True)  #We will remove these later

    else :  # Our date's data is not missing

        indexfound = binary_search(traffic['Date.Ordinal'], dateWanted)

        prevIndex = indexfound

    

        if indexfound > 0: # Good Case!

            SMA8col = SMA8col.append({'SMA_8': traffic['SMA_8'][indexfound]},ignore_index=True)

        else : # Bad Case: NOT FOUND 

            # Seek a match for timedate - timdelta(minutes=1)

            last = prevIndex

            prevMinute = dateWanted + timedelta(minutes=-1)

            prevIndex = binary_search(traffic['Date.Ordinal'], prevMinute)

            

            while prevIndex < 1 and k < 6:

                prevMinute = dateWanted + timedelta(minutes=minOffset[k])

                prevIndex = binary_search(traffic['Date.Ordinal'], prevMinute)

                k = k+1

                # The code below outputs a trace. This is how I found 2017-07-07 and 2017-03-10 to have missing data.

                #print(dateWanted)

                #print(last)

        

            SMA8col = SMA8col.append({'SMA_8': traffic['SMA_8'][prevIndex]},ignore_index=True)
newtrain = pd.concat([traindb,SMA8col],axis=1)
len(newtrain[newtrain['SMA_8']==0])
len(newtrain[newtrain['SMA_8']==0])/len(newtrain)
newtrain = newtrain[newtrain['SMA_8']!=0]
testdb = pd.read_csv('../input/canssi-ncsc-ferry-delays/test.csv')
testdb.head()
testdb['Full.Date'] = pd.to_datetime(testdb['Full.Date'] + ' ' + testdb['Scheduled.Departure'])

testdb.drop(['Scheduled.Departure','Day','Month', 'Day.of.Month', 'Year'],axis=1,inplace=True)
testdb.sort_values(by='Full.Date',inplace=True)
SMA8col_test = pd.DataFrame(columns=['SMA_8'])

for i in range(0,len(testdb)):

    dateWanted = testdb['Full.Date'][i]

    k=0

    minOffset = [1,-2,2,-3,3,-60,60] # I had to add -60,60 to accomodate traffic gap in data

    # Consider that traindb has departures before capture of traffic data began

    # We assign a 1 in traffic sma for this time period

    if dateWanted < traffic['Date.Ordinal'][0]:  # If the datetime is before data capture beginning date

        SMA8col_test = SMA8col_test.append({'SMA_8': 1},ignore_index=True)  

    elif (dateWanted.year == 2017 and dateWanted.month == 7 and dateWanted.day == 7) or (dateWanted.year == 2017 and dateWanted.month == 3 and dateWanted.day == 10):

        SMA8col_test = SMA8col_test.append({'SMA_8': 0},ignore_index=True) # This 0 value will help remove these days from our data soon. 

    else :  # Our date's data is not missing

        indexfound = binary_search(traffic['Date.Ordinal'], dateWanted)

    

        if indexfound > 0: # Good Case!

            SMA8col_test = SMA8col_test.append({'SMA_8': traffic['SMA_8'][indexfound]},ignore_index=True)

        else : # Bad Case: NOT FOUND 

            # Seek a match for timedate - timdelta(minutes=1)

            last = prevIndex

            prevMinute = dateWanted + timedelta(minutes=-1)

            prevIndex = binary_search(traffic['Date.Ordinal'], prevMinute)

            

            while prevIndex < 1 and k < 7:

                prevMinute = dateWanted + timedelta(minutes=minOffset[k])

                prevIndex = binary_search(traffic['Date.Ordinal'], prevMinute)

                k = k+1

                # The code below outputs a trace. This is how I found 2017-07-07 and 2017-03-10 to have missing data.

                #print("We want",dateWanted)

                #print(last)

                #print("LastMin:", prevMinute)

                #print("-------------------")

        

            SMA8col_test = SMA8col_test.append({'SMA_8': traffic['SMA_8'][prevIndex]},ignore_index=True)
newtest = pd.concat([testdb,SMA8col_test],axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import brier_score_loss
newtrain.head()
vessels = pd.get_dummies(newtrain['Vessel.Name'])

days = pd.get_dummies(newtrain['Day'])

trips = pd.get_dummies(newtrain['Trip'])
newtrain = pd.concat([newtrain,vessels,days,trips],axis=1)
newtrain.columns
X = newtrain[['SMA_8','Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',

       'Wednesday','Bowen Queen', 'Coastal Celebration', 'Coastal Inspiration',

       'Coastal Renaissance', 'Mayne Queen', 'Queen of Alberni',

       'Queen of Capilano', 'Queen of Coquitlam', 'Queen of Cowichan',

       'Queen of Cumberland', 'Queen of New Westminster',

       'Queen of Oak Bay', 'Queen of Surrey', 'Salish Eagle', 'Salish Raven',

       'Skeena Queen','Spirit of Vancouver Island'

       ]]

y = newtrain['Delay.Indicator']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
pred = logmodel.predict_proba(X_test)
pred = pred[:,1]
pd.DataFrame(pred).describe()
fpr, tpr, thresholds = roc_curve(y_test, pred)

plt.figure(figsize=(8,6))

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr)

plt.show()
roc_auc_score(y_test, pred)
newtest.head()
DayDict = {0: 'Monday',1: 'Tuesday', 2: 'Wednesday',3: 'Thursday',4: 'Friday',5:'Saturday', 6:'Sunday'}
DayCol = pd.DataFrame()
for i in range(0,len(newtest)):

    i_date = newtest['Full.Date'][i].weekday()

    DayCol = DayCol.append(pd.Series(DayDict[i_date]),ignore_index=True)
DayCol = pd.DataFrame(DayCol[0])

DayCol.columns = ['Day']
testvessel = pd.get_dummies(newtest['Vessel.Name'])

testday = pd.get_dummies(DayCol['Day'])
newtest = pd.concat([newtest,testvessel,testday],axis=1)
testtrip = pd.get_dummies(newtest['Trip'])

newtest = pd.concat([newtest,testtrip],axis=1)
newtest
X_pred = newtest[['SMA_8','Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',

       'Wednesday','Bowen Queen', 'Coastal Celebration', 'Coastal Inspiration',

       'Coastal Renaissance', 'Mayne Queen', 'Queen of Alberni',

       'Queen of Capilano', 'Queen of Coquitlam', 'Queen of Cowichan',

       'Queen of Cumberland', 'Queen of New Westminster',

       'Queen of Oak Bay', 'Queen of Surrey', 'Salish Eagle', 'Salish Raven',

       'Skeena Queen','Spirit of Vancouver Island'

       ]]
realPred = logmodel.predict_proba(X_pred)
lateProb = realPred[:,1]
submission = pd.concat([pd.DataFrame(lateProb),newtest['ID']],axis=1)
submission.columns = ['Delay.Indicator','ID']
submission.to_csv('LogReg_predictions',index=False)
from catboost import CatBoostClassifier
ctb = CatBoostClassifier(

    iterations=300,

    learning_rate=0.3,

    loss_function='Logloss')

ctb.fit(X,y,eval_set=(X_train,y_train),use_best_model=True)
ctbpred = ctb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, ctbpred)

plt.figure(figsize=(8,6))

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr)

plt.show()
roc_auc_score(y_test,ctbpred)
ctb_testpred = ctb.predict_proba(X_pred)[:,1]
ctbsubmission = pd.concat([pd.DataFrame(ctb_testpred),newtest['ID']],axis=1)
ctbsubmission.columns = ['Delay.Indicator','ID']
ctbsubmission.to_csv('catboost_predictions',index=False)