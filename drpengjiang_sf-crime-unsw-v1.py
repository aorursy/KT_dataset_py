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
#Step1:  Load datasets

sf_trainname= 'sf_train.csv'
sf_testname = 'sf_test_unlabelled.csv'

sf_train = pd.read_csv('../input/' + sf_trainname)
sf_test = pd.read_csv('../input/' + sf_testname)
print ("Read training dataset from %s yielded %d rows and %d columns" % (sf_trainname, sf_train.shape[0], sf_train.shape[1])) 
print ("Read test dataset from %s yielded %d rows and %d columns" % (sf_testname, sf_test.shape[0], sf_test.shape[1]))
sf_train.sample(5)
sf_test.sample(5)
#Step 2: Data exploration

list_num = ['DayOfMonth', 'Month',  'Year', 'TimeBin', 'X', 'Y'] 
list_cat = ['DayOfWeek', 'PdDistrict']

sf_train[list_num].hist()
# Step 3: Step pre-processing

dummy_feature = pd.get_dummies(sf_train[list_cat])
list_dummy_columns= list(dummy_feature.columns.values)
merge_sf = pd.concat([sf_train, dummy_feature], axis=1)

def get_mergesf(sf_train):
    
    dummy_feature = pd.get_dummies(sf_train[list_cat])
    list_dummy_columns= list(dummy_feature.columns.values)
    merge_sf = pd.concat([sf_train, dummy_feature], axis=1)
    
    return merge_sf
merge_sf.sample(5)
list_columns = list_dummy_columns + list_num
# Step 4: Train a machine learning model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as rfc
clf = rfc()
clf.fit(merge_sf[list_columns], merge_sf['target_crime'])
# Step 5: validation on the test set
merge_sf_test = get_mergesf(sf_test)
predict_result = clf.predict(merge_sf_test[list_columns])
# Step 6: write results to sf_test and send it for scoring

player = 'PJ'

sf_test['predict_result'] = predict_result

sf_test.to_csv( player+'_sf_test_labelled.csv', header = True )
