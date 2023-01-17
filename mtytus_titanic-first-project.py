#data analysis

import pandas as pd

import numpy as np

#visualization

import matplotlib.pyplot as plt

import seaborn as sns

#Machine learing

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv('../input/train.csv')

test_df  = pd.read_csv('../input/test.csv') 
#values

print(train_df.columns.values)



#head

train_df.head()