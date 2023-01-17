"""Initial exploratory analysis"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Constants

PATH = '../input/'

TRAIN = 'train.csv'

TEST = 'test.csv'



# Load Data

train_df = pd.read_csv(PATH + TRAIN)



print('Train Set')

train_df.head()
# Simple Metrics of Dataset

print('Number of examples: {}'.format(train_df.shape[0]))

print('Number of Features: {}'.format(train_df.shape[1] - 2))



# Distribution of target variable

print('\nMean of target variable: {}'.format(train_df['y'].mean()))

print('Unbiased Variance of target variable {}'.format(train_df['y'].var()))

plt.figure(figsize=(12,8))

sns.distplot(train_df['y'].values, bins=50, kde=False)

plt.xlabel('y variable', fontsize=12)

plt.ylabel('Frequency')

plt.show()
# Feature types and distributions

print('Feature Types and #')

print(train_df.dtypes.value_counts())



# Categorical Features

categoricals = train_df.columns[train_df.dtypes == object]

print('\nCategorical Features:')

print(categoricals.values,'\n')



# Let's Look at how many categories there are

for feature in categoricals:

    print('Feature {}: {} Categories'.format(str(feature), len(train_df[feature].unique())))
# Let's Now look at the values for the int64 features

int_features = train_df.columns[train_df.dtypes == 'int64']



values_dict = {}



for feature in int_features:

    values_dict[str(feature)] = len(train_df[feature].unique())



del values_dict['ID']

print('# Of unique Values for each int Feature')

print(values_dict)
drop = []

for key in values_dict:

    if values_dict[key] == 1:

        drop.append(key)

        

for feature in drop:

    print('Dropped Feature {}'.format(feature))

    del values_dict[feature]
# Loading Test Set



test_df = pd.read_csv(PATH + TEST)

print('TEST SET')

print(test_df.head())

print('shape = ',test_df.shape,'\n')



test_categoricals = test_df.columns[test_df.dtypes == object]



for feature in test_categoricals:

    print('Feature {}: {} Categories'.format(str(feature), len(test_df[feature].unique())))
for feature in categoricals:

    test_feature = test_df[feature].unique()

    train_feature = train_df[feature].unique()

    union = pd.Series(test_df[feature].tolist() + train_df[feature].tolist()).unique()

    

    test_feature.sort()

    train_feature.sort()

    union.sort()

    

    print('\n\nTest {}: {}'.format(feature,test_feature))

    print('\nTrain {}: {}'.format(feature,train_feature))

    print('\nUnion size: ',len(union))

    

    

    

    