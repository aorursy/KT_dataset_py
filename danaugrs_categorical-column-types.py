import numpy as np

import pandas as pd

from collections import Counter, defaultdict



dataTrain = pd.read_csv("../input/train_categorical.csv", chunksize=100000, dtype=str, usecols=list(range(1,2141)))

dataTest = pd.read_csv("../input/test_categorical.csv", chunksize=100000, dtype=str, usecols=list(range(1,2141)))
trainingCounts = defaultdict(Counter)

for idx, chunk in enumerate(dataTrain):

    for col in chunk:

        trainingCounts[col] += Counter(chunk[col].values)

    #print('Done with chunk {0}/12'.format(idx+1))
testCounts = defaultdict(Counter)

for idx, chunk in enumerate(dataTest):

    for col in chunk:

        testCounts[col] += Counter(chunk[col].values)

    #print('Done with chunk {0}/12'.format(idx+1))
emptyColumnsTest = [col for col in testCounts if len(list(testCounts[col].keys())) == 1]

binaryColumnsTest = [col for col in testCounts if len(list(testCounts[col].keys())) == 2]

multiColumnsTest = [col for col in testCounts if len(list(testCounts[col].keys())) > 2]



emptyColumnsTrain = [col for col in trainingCounts if len(list(trainingCounts[col].keys())) == 1]

binaryColumnsTrain = [col for col in trainingCounts if len(list(trainingCounts[col].keys())) == 2]

multiColumnsTrain = [col for col in trainingCounts if len(list(trainingCounts[col].keys())) > 2]



print('{0:5} | {1:^5} | {2:^7} | {3:^5}'.format('', 'Empty', 'Binary', 'Multi'))

print('{0:5} | {1:^5} | {2:^7} | {3:^5}'.format('Test', len(emptyColumnsTest), len(binaryColumnsTest), len(multiColumnsTest)))

print('{0:5} | {1:^5} | {2:^7} | {3:^5}'.format('Train', len(emptyColumnsTrain), len(binaryColumnsTrain), len(multiColumnsTrain)))
import seaborn as sns

%matplotlib inline



sns.set_style("whitegrid")

sns.set_context("talk")

sns.barplot(y='Value', x='Type', hue='Dataset', data=pd.DataFrame([

    ['Test', 'Empty', len(emptyColumnsTest)],

    ['Test', 'Binary', len(binaryColumnsTest)],

    ['Test', 'Multi-value', len(multiColumnsTest)],

    ['Train', 'Empty', len(emptyColumnsTrain)],

    ['Train', 'Binary', len(binaryColumnsTrain)],

    ['Train', 'Multi-value', len(multiColumnsTrain)]],

    columns=['Dataset', 'Type', 'Value'])).set(xlabel='', ylabel='Value')
sns.set_context("talk")

sns.heatmap(pd.DataFrame([

    [len(set(emptyColumnsTest).intersection(set(emptyColumnsTrain))),

     len(set(emptyColumnsTest).intersection(set(binaryColumnsTrain))),

     len(set(emptyColumnsTest).intersection(set(multiColumnsTrain)))],

    [len(set(binaryColumnsTest).intersection(set(emptyColumnsTrain))),

     len(set(binaryColumnsTest).intersection(set(binaryColumnsTrain))),

     len(set(binaryColumnsTest).intersection(set(multiColumnsTrain)))],

    [len(set(multiColumnsTest).intersection(set(emptyColumnsTrain))),

     len(set(multiColumnsTest).intersection(set(binaryColumnsTrain))),

     len(set(multiColumnsTest).intersection(set(multiColumnsTrain)))]],

    columns=['Train Empty', 'Train Binary', 'Train Multi-value'],

    index=['Test Empty', 'Test Binary', 'Test Multi-value']),

            annot=True, fmt="d", linewidths=.5)
trulyEmpty = set(emptyColumnsTest).intersection(set(emptyColumnsTrain))

trulyMulti = set(multiColumnsTest).union(set(multiColumnsTrain))

trulyBinary = set(binaryColumnsTest).union(set(binaryColumnsTrain)) - trulyMulti



len(trulyEmpty) + len(trulyBinary) + len(trulyMulti)
df=pd.DataFrame([

    ['Empty', 'Empty', len(set(emptyColumnsTest).intersection(set(emptyColumnsTrain)))],

    ['Empty', 'Binary', len(set(emptyColumnsTest).intersection(set(binaryColumnsTrain)))],

    ['Empty', 'Multi-value', len(set(emptyColumnsTest).intersection(set(multiColumnsTrain)))],

    ['Binary', 'Empty', len(set(binaryColumnsTest).intersection(set(emptyColumnsTrain)))],

    ['Binary', 'Binary', len(set(binaryColumnsTest).intersection(set(binaryColumnsTrain)))],

    ['Binary', 'Multi-value', len(set(binaryColumnsTest).intersection(set(multiColumnsTrain)))],

    ['Multi-value', 'Empty', len(set(multiColumnsTest).intersection(set(emptyColumnsTrain)))],

    ['Multi-value', 'Binary', len(set(multiColumnsTest).intersection(set(binaryColumnsTrain)))],

    ['Multi-value', 'Multi-value', len(set(multiColumnsTest).intersection(set(multiColumnsTrain)))]],

    columns=['Test', 'Train', 'Value'])
sns.set_palette('deep')

g = sns.barplot(y='Value', x='Train', hue='Test', data=df).set(ylabel='')

g[0].figure.suptitle('How column types from TRAINING data change when looking at TEST data')
g = sns.barplot(y='Value', x='Test', hue='Train', data=df).set(ylabel='')

g[0].figure.suptitle('How column types from TEST data change when looking at TRAINING data')