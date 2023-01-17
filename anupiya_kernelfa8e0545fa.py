# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

def getfloat(s): # convert the string value to float value

    try:

        return float(s) # convert the float value

    except ValueError:

        return 0.0 # if cannot convert the float value 0.0 will be returned

dataset2=[] # predefined array to save the dataset 

dataframe = read_csv('../input/MiningProcess_Flotation_Plant_Database.csv', engine='python') # reading the dataset

dataset = dataframe.values # extracting the data

dataset = dataset[:,1:] # skipping the date

for X in dataset: # accessing each data row in the dataset

    dataset1=[]

    for item in X: # accessing each item of the dataset row

        dataset1.append(getfloat(str(item).replace(',','.'))) # replacing the comma value by '.' to ease the convertion to float

    dataset2.append(dataset1) # creating the final dataset

scaler = MinMaxScaler(feature_range=(0,1)) # scaler created from 0-1

dataset = scaler.fit_transform(dataset2) #scaling the dataset

# creating the training and testing datasets uses only 80% of the data to train and 20% to test

train_size = int(len(dataset2) *0.8)

test_size = len(dataset2) - train_size

train_data = dataset2[:train_size]

test_data =dataset2[train_size:]

train_X = np.array(train_data)[:,:20] # accessing all the parameters 

train_Y = np.array(train_data)[:,20:21] # learning to predict the iron concerntration

test_X = np.array(test_data)[:,:20] 

test_Y = np.array(test_data)[:,20:21] # testing to predict the iron concerntration





# Any results you write to the current directory are saved as output.