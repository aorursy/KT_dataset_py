import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/train.csv')
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../input/test.csv").values

print (dataset.head())

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)
pred = rf.predict(test)
