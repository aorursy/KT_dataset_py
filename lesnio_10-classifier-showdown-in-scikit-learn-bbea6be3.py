

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


le = LabelEncoder().fit(train.species) 

print (le)

labels = le.transform(train.species)    # encode species strings

print (labels)



classes = list(le.classes_)    # save column names for submission

print (classes)

test_ids = test.id                             # save test ids for submission

    

train = train.drop(['species', 'id'], axis=1)  

test = test.drop(['id'], axis=1)

    
