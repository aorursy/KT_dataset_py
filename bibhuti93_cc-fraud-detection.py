# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print("Supplied dataset :",os.listdir("../input"))
#Storing the csv file contents in a dataset
dataset = pd.read_csv('../input/creditcard.csv')
# Any results you write to the current directory are saved as output.
#Print dataset first 10 rows
print(dataset.head(10))

#Print dataset size
print(len(dataset.index))
#Add a column "Transaction_ID" at index 0
dataset.insert(0, 'Transaction_ID', range(1, 1 + len(dataset.index)))
#Print dataset column headers
print(dataset.columns)
#Print number of columns in teh given dataset
print("Number of columns: ",len(dataset.columns))
#Detect NaN values
print(dataset.isnull().sum())
from sklearn.model_selection import train_test_split #Usd to spilt datase into test and training datasets
#Spilt dataset into train and test dataset based in ratio 80:20
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2)

#Print the train and test datasets
print("----------Train dataset---------")
print(train_dataset.head(5))
print("----------Test dataset---------")
print(test_dataset.head(5))

#Print training and test dataset size
print(len(train_dataset.index))
print(len(test_dataset.index))
#Copy test_dataset to validation dataset as we will need validation dataset to compare with prediction
validation_dataset = test_dataset
print(validation_dataset.head(5))
#Drop Class column from test_dataset because that is what we are going to predict
test_dataset = test_dataset.drop(columns=['Class'])
print(test_dataset.head(5))
#Prepare labels and features
train_dataset_feature = train_dataset.iloc[:,0:31]
train_dataset_label = train_dataset.iloc[:,-1]
print(train_dataset_feature.head(3))
print(train_dataset_label.head(3))
#Prepare GaussianNB model
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(train_dataset_feature,train_dataset_label)
prediction= gnb.predict(test_dataset)

#Preparing prediction dataset
prediction = pd.DataFrame({'Transaction_ID':test_dataset['Transaction_ID'],'Class':prediction})

#Saving to CSV
prediction = prediction.to_csv('cc_fraud_prediction.csv', index=False)

#Getting accuracy
accuracy = round(gnb.score(train_dataset_feature, train_dataset_label) * 100, 2)
print(accuracy)
