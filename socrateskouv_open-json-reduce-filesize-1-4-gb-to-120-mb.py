import numpy as np 
import pandas as pd 
import os
from pandas.io.json import json_normalize,loads
import time

def load_json(d):
    return loads(d)

def json_into_dataframe(dataframe,column):
    return json_normalize(dataframe[column].apply(load_json).tolist()).add_prefix(column +'.')
        
def open_flat(filepath,columns):
    counter = 0 
    data= pd.read_csv(filepath, low_memory = False)
    for column in columns :
        print ('Unpacking ' + column)
        temp = json_into_dataframe(dataframe = data, column = column)
        for item in temp.columns:
            if len(temp[item].unique()) == 1: # if a column has the same value for all rows is not significant
                temp.drop(item, axis = 1, inplace = True)
                print ('column '+ item + ' was dropped')
                counter += 1
        data = pd.concat([data,temp], axis = 1)
        data.drop([column],inplace = True, axis = 1)
    print ('Columns dropped :',counter)
    return data
train_path = "../input/train.csv"
test_path = "../input/test.csv"
columns = ['totals','device','geoNetwork','trafficSource']

t0 = time.time()
train = open_flat(filepath = train_path ,columns = columns)
t1 = time.time()
print ('time to load train set', t1-t0)
t0 = time.time()
test = open_flat(filepath = test_path ,columns = columns)
t1 = time.time()
print ('time to load test set ', t1-t0)
print(train.shape)
print(test.shape)
for column in train.columns:
    if column not in test.columns:
        print(column)
train.drop('trafficSource.campaignCode' , axis = 1 , inplace = True)
print(train.shape)
print(test.shape)
t0 = time.time()
writer = pd.ExcelWriter('train.xlsx')
train.to_excel(writer,'Sheet1')
writer.save()
t1 = time.time()
print ('time to save train dataset to excel ', (t1-t0)/60.0 , ' mins') 
t0 = time.time()
writer = pd.ExcelWriter('test.xlsx')
test.to_excel(writer,'Sheet1')
writer.save()
t1 = time.time()
print ('time to save test dataset to excel ', (t1-t0)/60.0 , ' mins') 
print ('File size of flat train set :' + str(((os.path.getsize("train.xlsx")/1024)/1024)) + ' MB')
print ('File size of original train set :' + str(((os.path.getsize("../input/train.csv")/1024)/1024)) + ' MB')
print ('File size of flat test set :' + str(((os.path.getsize("test.xlsx")/1024)/1024)) + ' MB')
print ('File size of original test set :' + str(((os.path.getsize("../input/test.csv")/1024)/1024)) + ' MB')