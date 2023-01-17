#The newly uploaded files : train_v2.csv and test_v2.csv are very large in size(approx. 24 GB) 

#which was shooting up my laptop memory. Hence I have worked with earlier loaded files containing less data





import pandas as pd #FOR DATA PRE-PROCESSING

import json #FOR CONVERTING STRING TYPE DICT VALUES TO PYTHON DICT OBJECT

import os



print(os.listdir())

print(os.getcwd())





########### LOADING CSV FILE ############

train = pd.read_csv('../input/train.csv')

#########################################



########## CONVERTING STRING TYPE DICT VALUES TO PYTHON DICT OBJECT ###########

print('Before converting data to dict, datatype is --->',type(train.loc[0,'device']))

dictColumns = ['device','geoNetwork','totals','trafficSource']

for cols in dictColumns:

    train[cols] = train[cols].apply(json.loads)

print('After converting data to dict, datatype is --->',type(train.loc[0,'device']))

###############################################################################





######### CREATING NEW COLUMN FOR EACH KEY FROM DICT DATA #############

print('Before Creating New Columns, shape of dataframe is--->',train.shape)   

for cols in dictColumns:

    train = pd.concat([train.drop(cols,axis=1), train[cols].apply(pd.Series)], axis=1)

print('After Creating New Columns, shape of dataframe is--->',train.shape)

#######################################################################






























