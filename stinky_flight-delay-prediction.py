# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import linregress

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/flight-delay-prediction/Jan_2019_ontime.csv')

df.drop(['Unnamed: 21'], axis=1, inplace = True)
df.head()
df.info()
dflinreg=df

dflinreg.dropna(inplace = True)

linregress(dflinreg['DEP_DEL15'], dflinreg['ARR_DEL15'])
linregress(dflinreg['DISTANCE'], dflinreg['ARR_DEL15'])
dftemp=df[['DIVERTED','DEP_DEL15']].copy()

dftemp.dropna(inplace=True)

linregress(dftemp['DIVERTED'], dftemp['DEP_DEL15'])
dftemp[['DIVERTED','DEP_DEL15']].groupby(['DIVERTED']).mean()
delayPerAirport=df[['DEP_DEL15','ORIGIN_AIRPORT_ID']].groupby(['ORIGIN_AIRPORT_ID']).mean()

delayPerAirport.reset_index(inplace=True)

plt.scatter(delayPerAirport['ORIGIN_AIRPORT_ID'],delayPerAirport['DEP_DEL15'])
delayPerAirport=df[['ARR_DEL15','DEST_AIRPORT_ID']].groupby(['DEST_AIRPORT_ID']).mean()

delayPerAirport.reset_index(inplace=True)

plt.scatter(delayPerAirport['DEST_AIRPORT_ID'],delayPerAirport['ARR_DEL15'])
delayPerAirport=df[['ARR_DEL15','ORIGIN_AIRPORT_ID']].groupby(['ORIGIN_AIRPORT_ID']).mean()

delayPerAirport.reset_index(inplace=True)

plt.scatter(delayPerAirport['ORIGIN_AIRPORT_ID'],delayPerAirport['ARR_DEL15'])
delayDepPerTimeSlot= df[['DEP_TIME_BLK', 'DEP_DEL15']].groupby(['DEP_TIME_BLK']).mean()

delayDepPerTimeSlot
#Function to transfer the arrival time to an arrival time block

def timeToBlock(t):

    block="Nan"

    if(t> 0 and t< 600): block="0001-0559"

    if(t>559 and t< 700): block= "0600-0659"

    if(t>659 and t< 800): block= "0700-0759"

    if(t>759 and t< 900): block= "0800-0859"

    if(t>859 and t< 1000): block= "0900-0959"

    if(t>959 and t< 1100): block= "1000-1059"

    if(t>1059 and t< 1200): block= "1100-1159"

    if(t>1159 and t< 1300): block= "1200-1259"

    if(t>1259 and t< 1400): block= "1300-1359"

    if(t>1359 and t< 1500): block= "1400-1459"

    if(t>1459 and t< 1600): block= "1500-1559"

    if(t>1559 and t< 1700): block= "1600-1659"

    if(t>1659 and t< 1800): block="1700-1759"

    if(t>1759 and t< 1900): block= "1800-1859"

    if(t>1859 and t< 2000): block= "1900-1959"

    if(t>1959 and t< 2100): block= "2000-2059"

    if(t>2059 and t< 2200): block= "2100-2159"

    if(t>2159 and t< 2300): block= "2200-2259"

    if(t>2259 and t< 2400): block="2300-2359"

    return block
df['ARR_TIME_BLK']=df['ARR_TIME'].apply(timeToBlock)

df.head()
delayArrPerTimeSlot= df[['ARR_TIME_BLK', 'ARR_DEL15']].groupby(['ARR_TIME_BLK']).mean()

delayArrPerTimeSlot
from sklearn.model_selection import train_test_split



X=df.drop(['ARR_DEL15','CANCELLED', 'OP_UNIQUE_CARRIER', 'TAIL_NUM','OP_CARRIER_FL_NUM'

           ,'DEST_AIRPORT_SEQ_ID','ARR_TIME_BLK', 'ORIGIN', 'OP_CARRIER', 'DEP_TIME_BLK',

          'DEST','ORIGIN_AIRPORT_SEQ_ID'], axis=1)

df.dropna(inplace=True)

#X=  X.select_dtypes(exclude=['object'])

y=df['ARR_DEL15']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



X.head()
df.isnull().any()
from sklearn.ensemble import RandomForestClassifier



clf= RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train,y_train)
clf.score(X_test, y_test)