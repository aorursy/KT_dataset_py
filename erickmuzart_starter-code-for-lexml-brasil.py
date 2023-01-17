#from mpl_toolkits.mplot3d import Axes3D

#from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(sorted(os.listdir('../input')))
len(os.listdir('../input/2019/data/json/2019'))
print(sorted(os.listdir('../input/2019/data/json/2019')))
# Read contents of one of the json files: 1_2019.json

df = pd.read_json("../input/2019/data/json/2019/1_2019.json")
df.columns
df.describe(include="object")
df.dtypes
# Column `data` should be of type date:

df['data2'] = pd.to_datetime(df['data'])
# Number of documents per month

df['data2'].groupby(df["data2"].dt.month).count().plot(kind="bar")
df['autoridade'].value_counts()
df['facet-localidade'].value_counts()
df['facet-tipoDocumento'].value_counts()