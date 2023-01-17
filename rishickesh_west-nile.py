# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sampleSubmission.csv')

weather = pd.read_csv('../input/weather.csv')
train.head()
test.head()
weather.head()
labels = train.WnvPresent.values



# Not using codesum for this benchmark

weather = weather.drop('CodeSum', axis=1)



# Split station 1 and 2 and join horizontally

weather_stn1 = weather[weather['Station']==1]

weather_stn2 = weather[weather['Station']==2]

weather_stn1 = weather_stn1.drop('Station', axis=1)

weather_stn2 = weather_stn2.drop('Station', axis=1)

weather = weather_stn1.merge(weather_stn2, on='Date')
weather = weather.replace('M', -1)

weather = weather.replace('-', -1)

weather = weather.replace('T', -1)

weather = weather.replace(' T', -1)

weather = weather.replace('  T', -1)
weather.head()
def create_month(x):

    return x.split('-')[1]



def create_day(x):

    return x.split('-')[2]



train['month'] = train.Date.apply(create_month)

train['day'] = train.Date.apply(create_day)

test['month'] = test.Date.apply(create_month)

test['day'] = test.Date.apply(create_day)



# Add integer latitude/longitude columns

train['Lat_int'] = train.Latitude.apply(int)

train['Long_int'] = train.Longitude.apply(int)

test['Lat_int'] = test.Latitude.apply(int)

test['Long_int'] = test.Longitude.apply(int)
train.head()
train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)

test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

train = train.merge(weather, on='Date')

test = test.merge(weather, on='Date')

train = train.drop(['Date'], axis = 1)

test = test.drop(['Date'], axis = 1)
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()



from sklearn.preprocessing import scale

colu=['Species','Street','Trap']



for col in colu:

    train[col] = le.fit_transform(train[col])
train.head()
train = train.ix[:,(train != -1).any(axis=0)]

test = test.ix[:,(test != -1).any(axis=0)]
train.head()
colu=['month', 'day', 'Tavg_x', 'Depart_x',

       'WetBulb_x', 'Heat_x', 'Cool_x', 'Sunrise_x', 'Sunset_x', 'Depth_x',

       'Tavg_y', 'WetBulb_y', 'Heat_y', 'Cool_y']

col1=['SnowFall_x','PrecipTotal_x', 'StnPressure_x', 'SeaLevel_x', 'AvgSpeed_x','PrecipTotal_y', 'StnPressure_y', 'SeaLevel_y','AvgSpeed_y']



for col in colu:

    train[col]=train[col].astype(str).astype(int)

for col in col1:

    train[col]=train[col].astype(str).astype(float)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( train, labels, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test) 
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

log=LogisticRegression(penalty='l2',C=.00001)

log.fit(X_train,y_train)

y_pred = log.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

score=f1_score(y_test,y_pred,average='weighted')

score,accuracy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10,oob_score=True ,random_state =42, min_samples_split=25)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import f1_score

score=f1_score(y_test,y_pred,average='weighted')

score,accuracy

from sklearn.svm import SVC

clf = SVC(random_state = 100, kernel='rbf')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import f1_score

score=f1_score(y_test,y_pred,average='weighted')

score,accuracy
train.head()
import xgboost as xgb

model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score=f1_score(y_test,y_pred,average='weighted')

score

accuracy = accuracy_score(y_test, y_pred)

score,accuracy
model = Sequential()

model.add(Dense(20, input_dim=42, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10,validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
scores = model.evaluate(X_test, y_test)

scores
labels=pd.DataFrame(labels)
labels.columns=['ind']

labels.head()
labels.ind.value_counts()
train.head()
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_resample(train, labels)



X_sm=pd.DataFrame(X_sm)
X_sm.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X_sm, y_sm, test_size=0.33, random_state=42)
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

log=LogisticRegression(penalty='l2',C=.00001)

log.fit(X_train,y_train)

y_pred = log.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

score=f1_score(y_test,y_pred,average='weighted')

score,accuracy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20,oob_score=True ,random_state =42, min_samples_split=25)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) 

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import f1_score

score=f1_score(y_test,y_pred,average='weighted')

score,accuracy

import xgboost as xgb

model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score=f1_score(y_test,y_pred,average='weighted')

score

accuracy = accuracy_score(y_test, y_pred)

score,accuracy
from sklearn.svm import SVC

clf = SVC(random_state = 100, kernel='rbf')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

accuracy

from sklearn.metrics import f1_score

score=f1_score(y_test,y_pred,average='weighted')

score,accuracy