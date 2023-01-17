# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from random import seed

from random import random

from random import randrange





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Create a random subsample from the dataset with replacement

def subsample(dataset, ratio=1.0):

    sample = list()

    n_sample = round(len(dataset) * ratio)

    while len(sample) < n_sample:

        index = randrange(len(dataset))

        sample.append(dataset[index])

    return sample





# Calculate the mean of a list of numbers

def mean(numbers):

    return sum(numbers) / float(len(numbers))

seed(1)

# True mean

dataset = [[randrange(10)] for i in range(10)]

print(dataset)

print('True Mean: %.3f' % mean([row[0] for row in dataset]))

# Estimated means

ratio = 0.5

for size in [1, 5, 10]:

    sample_means = list()

    for i in range(size):

        sample = subsample(dataset, ratio)

        print(sample)

        sample_mean = mean([row[0] for row in sample])

        sample_means.append(sample_mean)

    print('Samples=%d, Estimated Mean: %.3f' % (size, mean(sample_means)))