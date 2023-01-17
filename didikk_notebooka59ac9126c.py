import numpy as np

import pandas as pd 





#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train_dt = pd.read_csv('../input/train.csv')

print(train_dt.size)

print(train_dt.columns)
train_dt[1:10]
train_dt.info()
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

x = train_dt.Survived

y = train_dt.Age

pearsonr(x,y)
plt.scatter(y, x)