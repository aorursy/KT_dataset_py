import numpy as np 

import pandas as pd

import seaborn as sns

import pandas_profiling as pp

import matplotlib.pyplot as plt

from sklearn import decomposition

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/cjs-dataset/cjs.csv')

df.head()
df.info()
cats=df.select_dtypes(include=['object']).columns

cats=cats.drop(['TR'],1)

cats
labelencoder = LabelEncoder()

for c in cats:

    df[c] = labelencoder.fit_transform(df[c])
df.head()
pp.ProfileReport(df)