import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import csv

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X = data.values[:,1:].astype(float)
Y = data.values[:,0]

#%% run model

nTree = 1000
alg = RandomForestClassifier(nTree)

alg.fit(X,Y)
pre = alg.predict(test)



