# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Any results you write to the current directory are saved as output.
print (train.shape)

print (test.shape)
print (train.shape)

print (test.shape)