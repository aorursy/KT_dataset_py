%matplotlib inline

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

#import seaborn as sns
#lists the files in the folder

#import os

#print(os.listdir("data"))
#Reads in the csv-files and creates a dataframe using pandas



#train = pd.read_csv('data/housing_data.csv')

#test = pd.read_csv('data/housing_test_data.csv')

#sampleSubmission = pd.read_csv('data/sample_submission.csv')
train = pd.read_csv('../input/dat158-2019/housing_data.csv')

test = pd.read_csv('../input/dat158-2019/housing_test_data.csv')

sampleSubmission = pd.read_csv('../input/dat158-2019/sample_submission.csv')
train.head()
test.head()
train.info()
test.info()
train.describe()
median_house_value = [0 for i in test['Id']]
len(median_house_value)
median_house_value[:10]
submission = pd.DataFrame({'Id': test['Id'], 'median_house_value': median_house_value})
submission.head()
# Stores a csv file to submit to the kaggle competition

#submission.to_csv('submission.csv', index=False)