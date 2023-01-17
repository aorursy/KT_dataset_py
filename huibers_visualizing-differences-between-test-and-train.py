import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
pd.__version__
# read datasets, force object data type to preserve number formatting
train = pd.read_csv('../input/train.csv', dtype=object)
train.shape
test = pd.read_csv('../input/test.csv', dtype=object)
test.shape
# the most common values have a single significant figure for target
train.target.value_counts()
# remove ID and target columns, examine population of remaining values
# note there are two formats for zero, some values do not have a trailing ".0"
train.iloc[:,2:].stack().value_counts()
# remove ID column, examine population of values
# test data has high precision values not present in train
test.iloc[:,1:].stack().value_counts()
# flatten dataframe into a series
train_series = pd.Series(train.iloc[:,2:].astype(np.float64).values.flatten())
train_series.shape
# how many nonzero values?
train_series[train_series>0].shape[0]
# how sparse is the train dataset? 3.1%
train_series[train_series>0].shape[0]/train_series.shape[0]
test_series = pd.Series(test.iloc[:,1:].astype(np.float64).values.flatten())
test_series.shape
# how sparse is the test dataset? 1.4%, significantly less than train
test_series[test_series>0].shape[0]/test_series.shape[0]
# clean up to fit this notebook into Kaggle's 17 GB limit
del train_series
del test_series
# some test dataset statistics, look at nonzero values only
train_stats = train.ID.to_frame()
train_stats['count']  = train.iloc[:,2:].astype(np.float64).replace(0.0, np.nan).count(axis=1)
train_stats['unique'] = train.iloc[:,2:].astype(np.float64).replace(0.0, np.nan).nunique(axis=1)
# some test dataset statistics, look at nonzero values only
test_stats = test.ID.to_frame()
test_stats['count']  = test.iloc[:,1:].astype(np.float64).replace(0.0, np.nan).count(axis=1)
test_stats['unique'] = test.iloc[:,1:].astype(np.float64).replace(0.0, np.nan).nunique(axis=1)
# there are unusual clusters, a series of points with a negative slope, not seen in the test dataset
train_stats.plot.scatter(x='count', y='unique', figsize=(12,10), alpha=0.3)
# examine the lower left more closely, there are more of these clusters
train_stats[train_stats['count']<900].plot.scatter(x='count', y='unique', figsize=(12,10), alpha=0.3)
test_stats.plot.scatter(x='count', y='unique', figsize=(12,10), alpha=0.3)
# this cluster stands out in the test dataset
test_stats[test_stats['count'] == test_stats['unique']].plot.scatter(x='count', y='unique', alpha=0.3)
# how many 'customers' are in this count=unique set? 
test_stats[test_stats['count'] == test_stats['unique']].count()['ID']
# how many 'customers' are not in this count=unique set? 
test_stats[test_stats['count'] != test_stats['unique']].count()['ID']
# test the sparsity of the values in this count=unique set
test_series = pd.Series(test.iloc[:,1:].astype(np.float64).values.flatten())
test[test.ID.isin(test_stats[test_stats['count'] == test_stats['unique']]['ID'])].shape
test_series_count_unique = pd.Series(test[test.ID.isin(test_stats[test_stats['count'] == test_stats['unique']]['ID'])].iloc[:,1:].astype(np.float64).values.flatten())
# this count=unique set is very sparse, 0.70%
test_series_count_unique[test_series_count_unique>0].shape[0]/test_series_count_unique.shape[0]
test[test.ID.isin(test_stats[test_stats['count'] != test_stats['unique']]['ID'])].shape
test_series_count_not_unique = pd.Series(test[test.ID.isin(test_stats[test_stats['count'] != test_stats['unique']]['ID'])].iloc[:,1:].astype(np.float64).values.flatten())
# this count!=unique set less sparse, closer to the 3.1% of the train dataset
test_series_count_not_unique[test_series_count_not_unique>0].shape[0]/test_series_count_not_unique.shape[0]
