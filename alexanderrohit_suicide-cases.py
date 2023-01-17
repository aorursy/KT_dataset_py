import numpy as np

import pandas as pd

import os

import matplotlib as mpl
df=pd.read_csv("../input/who_suicide_statistics.csv")

df.head(10)
df.shape
df.corr()
df.axes
df.groupby(['year']).mean()

df.describe()

df.groupby(['country']).plot(kind='pie')