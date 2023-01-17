%matplotlib inline
%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

import numpy as np

import pandas as pd



# create the training & test sets, skipping the header row with [1:]

dataset = pd.read_csv("../input/train.csv")

target = dataset[[0]].values.ravel()

train = dataset.iloc[:,1:].values

test = pd.read_csv("../input/test.csv").values



# create and train the random forest

# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

rf = RandomForestClassifier(n_estimators=100)