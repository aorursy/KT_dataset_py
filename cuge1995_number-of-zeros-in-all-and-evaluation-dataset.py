import os

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

dg = df.iloc[:, -28:]
dg.head()
a = np.array((df == 0).astype(int).sum(axis=1))

df['zeros'] = a

df['percent'] = df['zeros']/1941
df['percent'].plot.hist()
a = np.array((dg == 0).astype(int).sum(axis=1))

dg['zeros'] = a

dg['percent'] = dg['zeros']/28
dg.head()
dg['percent'].plot.hist()