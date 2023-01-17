# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')

data.head()
data.isnull().sum()
print('We have {} features in our data'.format(len(data.columns)))
def entropy(labels):

    entropy=0

    label_counts = Counter(labels)

    for label in label_counts:

        prob_of_label = label_counts[label] / len(labels)

        entropy -= prob_of_label * math.log2(prob_of_label)

    return entropy



def information_gain(starting_labels, split_labels):

    info_gain = entropy(starting_labels)

    for branched_subset in split_labels:

        info_gain -= len(branched_subset) * entropy(branched_subset) / len(starting_labels)

    return info_gain
def split(dataset, column):

    split_data = []

    col_vals = data[column].unique() # This tree generation method only works with discrete values

    for col_val in col_vals:

        split_data.append(dataset[dataset[column] == col_val])

    return(split_data)
def find_best_split(dataset):

    best_gain = 0

    best_feature = 0

    features = list(dataset.columns)

    features.remove('class')

    for feature in features:

        split_data = split(dataset, feature)

        split_labels = [dataframe['class'] for dataframe in split_data]

        gain = information_gain(dataset['class'], split_labels)

        if gain > best_gain:

            best_gain, best_feature = gain, feature

    print(best_feature, best_gain)

    return best_feature, best_gain



new_data = split(data, find_best_split(data)[0]) # contains a list of dataframes after splitting