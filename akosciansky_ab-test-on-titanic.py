# Get the libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
# Import the data

train = pd.read_csv('../input/train.csv')
train.head()
# How much data is there? Is it balanced?

print(len(train))
print(len(train[train['Sex'] == 'male']))
print(len(train[train['Sex'] == 'female']))
# What are the survival rates for male and female?

train.groupby(by='Sex')['Survived'].mean()
# Use bootstrapping (500 samples with replacement) to get mor data points, which allows us to make any inference whether the survival rate for female is actually 
# higher than that for male

results = []

for i in range(500):
    sample = train.sample(frac=1, replace=True).groupby(by='Sex')['Survived'].mean()
    results.append(sample)

results = pd.DataFrame(results)
ax = results.plot.kde()
ax.set_xlabel('Survival Rate')
# The plot probably speaks for itself but let's still get the probability that female survival rate is higher than that for male. Usually we are looking for
# a threshold of 95%, which is the equivalent of a p-value of 0.05.

results['diff'] = 100 * (results['female'] - results['male']) / results['male']

prob = (results['diff'] > 0).mean()
print('The probability that female survivale rate is higher than that of male is: {:.1%}'.format(prob))
