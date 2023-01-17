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
#This kernel is the code for submission for competition "AI:challange Cognizance 2018"
#Dataset contains data point of various accidents and we were to determine the criticality of the accident that are unlabelled
train = pd.read_csv("../input/train.csv") 
train_Y = train['criticality']
train_X = train.drop(columns=['victim_id','criticality'])
train_X.head() #crude training features
train_X['incident_date'] = pd.to_datetime(train_X['incident_date']) #converted date to recognisable format
train_X['day_of_week'] = train_X['incident_date'].dt.weekday_name #used the day of week as important data
train_X = train_X.drop(columns=['incident_date'])
train_X.head() #training data which is categorical
time = pd.DatetimeIndex(train_X['incident_time'])
train_X['time'] = time.hour * 60 + time.minute #converted time to meaningful
train_X = train_X.drop(columns=['incident_time','incident_location']) #dropped incident location which is of lesser significance
train_X = pd.get_dummies(train_X) #converted categorical to numerical with the help of pandas
train_X.head()
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
clf = GaussianNB()
clf2 = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_X, train_Y) #used NB to train the classifier
clf2.fit(train_X, train_Y) #used RF Classifier
test_X = pd.read_csv('../input/test.csv') 
victim_id = test_X['victim_id']
test_X = test_X.drop(columns=['victim_id'])
test_X['incident_date'] = pd.to_datetime(test_X['incident_date'])
test_X['day_of_week'] = test_X['incident_date'].dt.weekday_name
test_X = test_X.drop(columns=['incident_date'])
time = pd.DatetimeIndex(test_X['incident_time'])
test_X['time'] = time.hour * 60 + time.minute
test_X = test_X.drop(columns=['incident_time','incident_location'])
test_X = pd.get_dummies(test_X)
test_X.head() #created synchronous test dataset
pred = clf2.predict(test_X)
output = pd.DataFrame(data={'victim_id':victim_id,'criticality':pred})
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.head()
output.to_csv( "submission.csv", index=False )

#Naive Bayes gave a decent score. Didn't get time to use RF classifier.