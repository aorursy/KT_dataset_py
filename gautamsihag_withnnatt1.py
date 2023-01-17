# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import datetime

train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000000)
test1 = pd.read_csv("../input/test.csv")
#train1test1 = train1test.ix[1000:,:]
#train1test1.info()
test1['date_time'] = pd.to_datetime(test1["date_time"])
test1['year'] = test1['date_time'].dt.year
test1['month'] = test1['date_time'].dt.month
test1['day_of_week'] = test1['date_time'].dt.dayofweek
test1['day'] = test1['date_time'].dt.day
test1['hour'] = test1['date_time'].dt.hour
train1['date_time'] = pd.to_datetime(train1["date_time"])
train1['year'] = train1['date_time'].dt.year
train1['month'] = train1['date_time'].dt.month
train1['day_of_week'] = train1['date_time'].dt.month
train1['day'] = train1['date_time'].dt.day
train1['hour'] = train1['date_time'].dt.hour
train1 = train1.drop('day_of_week', axis=1)
train1['day_of_week'] = train1['date_time'].dt.dayofweek
train1.ix[(train1['hour'] >= 10) & (train1['hour'] < 18), 'hour'] = 1
train1.ix[(train1['hour'] >= 18) & (train1['hour'] < 22), 'hour'] = 2
train1.ix[(train1['hour'] >= 22) & (train1['hour'] == 24), 'hour'] = 3
train1.ix[(train1['hour'] >= 1) & (train1['hour'] < 10), 'hour'] = 3
test1.ix[(test1['hour'] >= 10) & (test1['hour'] < 18), 'hour'] = 1
test1.ix[(test1['hour'] >= 18) & (test1['hour'] < 22), 'hour'] = 2
test1.ix[(test1['hour'] >= 22) & (test1['hour'] == 24), 'hour'] = 3
test1.ix[(test1['hour'] >= 1) & (test1['hour'] < 10), 'hour'] = 3
train1 = train1.fillna(-1)
test1 = test1.fillna(-1)
#train1test1 = train1test1.fillna(-1)
#train3 = train1
train1_b = train1[train1['is_booking'] == 1].drop('is_booking', axis=1)
hotelCluster = train1_b.ix[:,'hotel_cluster']
train1_b = train1_b.drop('hotel_cluster', axis=1) #df.drop('reports', axis=1)
test1_b.info()
train1_b.info()
train1_b = train1_b[['orig_destination_distance','srch_destination_id','srch_destination_type_id','year','month','day_of_week','hour']]
test1_b = test1[['orig_destination_distance','srch_destination_id','srch_destination_type_id','year','month','day_of_week','hour']]
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(15, 100), random_state=1)
clf.fit(train1_b, hotelCluster) 
y_probb=clf.predict_proba(test1_b)
y_probb[5130,:]