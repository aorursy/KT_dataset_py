import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
# I would like to see the behaviour of the values of each category

# look up how this was done in the redhat competition

print('category | range of values')

for ii in list(train.columns.values):

    print('{}: {}'.format(ii, len(set(train[ii]))))
# an alternative way to guage a highlevel overview

[train[ii].value_counts() for ii in list(train.columns.values)]
# ok some pretty small categories and some big ones

# post cat98 seems to go funky

# TODO figure out an interesting way to see what contains what in a quick glance

    # IE output what kable does in R

print('category | range of values | examples')

for ii in list(train.columns.values):

    print('{} | {} | {}, {}, {} ...'.format(ii, len(set(train[ii])), train[ii][0], train[ii][1], train[ii][2]))
## structure of loss

def sum_stats(col):

    print('max loss: %.2f' % col.max())

    print('min loss: %.2f' % col.min())

    print('median loss %.2f' % col.median())

    print('mean loss %.2f' % col.mean())

    print('std loss %.2f' % col.std())



# long tail 

sum_stats(train['loss'])
# plot the loss function out

train['loss'].hist(bins=100).set_xlim(0, 20000)

plt.ylabel('Number of claims')

plt.xlabel('Loss Bucket')

plt.title('loss distribution'.title())
# These are special kinds of events



# these are the kinds of losses that could really bring a business down (or loss a competition)

len(train[train['loss'] > 20000])
# Should we look at log loss?

train['log_loss'] = np.log(train['loss'])

train['log_loss'].hist(bins=50)

sum_stats(train['log_loss'])

plt.ylabel('Number of Claims')

plt.xlabel('Log Loss Bucket')

plt.title('log loss distribution'.title())



# Looks much more normally distributed
train['cont1'].hist()