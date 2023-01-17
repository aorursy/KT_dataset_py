#Imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import LabelEncoder

import os

import warnings
warnings.filterwarnings('ignore')
#Get a list of file names
print(os.listdir('../input/'))
#Initialize training data
train = pd.read_csv('../input/application_train.csv')
train.set_index('SK_ID_CURR', inplace=True)
train.head()
#Initialize testing data
test = pd.read_csv('../input/application_test.csv')
test.set_index('SK_ID_CURR', inplace=True)
test.head()
#The first thing I like to do is look at the state of the data, examining missing values and the like.
def gen_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_percentages = (100 * df.isnull().sum() / len(df)).round(1)
    missing_values_table = pd.concat([missing_values, missing_values_percentages], axis = 1)
    missing_values_table = missing_values_table.rename(columns = {0: 'Missing Values', 1: 'Percentage'})
    missing_values_table.sort_values('Missing Values', ascending=False, inplace=True)
    return missing_values_table

missing_values = gen_missing_values(train)
missing_values.head()
#I'll probably have to figure out some way to impute this data later, but for now I'll leave it.
#I imagine dealing with non-decimal values is a more pressing concern.
#The debate comes into play with label encoding vs. one-hot encoding and I chose to employ the safest approach which is to label encode all binary columns while 
#one-hot encoding all non-binary columns.

def adjust_for_labels(df): 
    le = LabelEncoder()
    for col in df:
        if df[col].dtype == 'object':
            dropped_df = df.dropna(subset=[col])
            if len(list(dropped_df[col].unique())) <= 2:
                le.fit(dropped_df[col])
                dropped_df[col] = le.transform(dropped_df[col])
                df.update(dropped_df[col])
    return df

train = adjust_for_labels(train)   
test = adjust_for_labels(test)

#One hot-encoding
train = pd.get_dummies(train)
test = pd.get_dummies(test)

#Let's also check the shape of the training and testing data as the features in both these sets must be the same.
print('Testing shape: ', test.shape)
print('Training shape: ', train.shape)
#As we can see, the one-hot encoding has most likely created more columns in the training data rather than the testing data (due to the larger amount of data,
#hence higher potential for a larger diversity of categorical labels)

#This is the extra column in the training data we need to save before we align the dataframes to each other
train_labels = train['TARGET']

train, test = train.align(test, join="inner", axis=1)

#Add this column back in
train["TARGET"] = train_labels

#Let's check the shape again
print('Testing shape: ', test.shape)
print('Training shape: ', train.shape)
#Nice.
#The next thing on the list is to identify outliers inside the data, and for this, I'm going to use the z-score, and set an artbitrary 
#threshold value of 3 std. deviations to locate potential problem columns, and just take a look at how the data plays out on those.

def compute_potential_anomalies(df): 
    anomaly_table = df.dropna()
    for col in anomaly_table:
        std = anomaly_table[col].std()
        anomaly_table.apply(lambda x: (x > 3 * std), axis=1)
    return anomaly_table

anomalies = compute_potential_anomalies(train)
anomalies.head()
        
