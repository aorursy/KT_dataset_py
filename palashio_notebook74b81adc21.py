# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import sklearn as sk

from sklearn import datasets, linear_model







trainingSet = pd.read_csv("../input/train.csv")

testSet = pd.read_csv("../input/test.csv")



del trainingSet['dropoff_datetime']

del trainingSet['store_and_fwd_flag']

del trainingSet['vendor_id']



trainingSet.columns = ['id', 'datePickup', 'passengerCount', 'pickupLongitude', 'pickupLatitude', 'dropoffLongitude', 'dropoffLatitude', 'duration']



trainingSet['timePickup'] = trainingSet['datePickup'] 

trainingSet['datePickup'] = trainingSet['datePickup'].apply(lambda x: x.split(' ')[0])

trainingSet['timePickup'] = trainingSet['timePickup'].apply(lambda x: x.split(' ')[1])

trainingSet['timePickup'] = trainingSet['timePickup'].apply(lambda x: x.split(':')[1])

trainingSet['datePickup'] = pd.to_datetime(trainingSet['datePickup'])

trainingSet['datePickup'] = trainingSet['datePickup'].dt.dayofweek



#****************************************************



del testSet['store_and_fwd_flag']

del testSet['vendor_id']





testSet.columns = ['id', 'datePickup', 'passengerCount', 'pickupLongitude', 'pickupLatitude', 'dropoffLongitude', 'dropoffLatitude']



testSet['timePickup'] = testSet['datePickup'] 

testSet['datePickup'] = testSet['datePickup'].apply(lambda x: x.split(' ')[0])

testSet['timePickup'] = testSet['timePickup'].apply(lambda x: x.split(' ')[1])

testSet['monthPickup'] = testSet['datePickup'] 

testSet['datePickup'] = pd.to_datetime(testSet['datePickup'])

testSet['datePickup'] = testSet['datePickup'].dt.dayofweek

testSet['monthPickup'] = testSet['monthPickup'].apply(lambda x: x.split('-')[1])

testSet['monthPickup'] = testSet['monthPickup'].apply(lambda x: x.split('0')[1])

testSet['timePickup'] = testSet['timePickup'].apply(lambda x: x.split(':')[0])



featureSet = trainingSet.copy()

del featureSet['duration']

del featureSet['id']



labelSet = trainingSet['duration']



del testSet['id']









regr = linear_model.LinearRegression()

regr.fit(featureSet, labelSet)

#regr.predict(testSet)



print (featureSet - testSet)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


