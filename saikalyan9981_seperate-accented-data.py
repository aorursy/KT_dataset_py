from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
nRowsRead = None # specify 'None' if want to read whole file

# cv-invalid.csv has 25404 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/cv-valid-dev.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'cv-valid-dev.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns in {df1.dataframeName}')

df2 = pd.read_csv('../input/cv-valid-test.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'cv-valid-test.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns in {df2.dataframeName}')
# df1.head(5)

accent_true = (df1["accent"].notna())

df_accented_dev = df1[accent_true]

print(df_accented_dev.shape)

df_accented_dev.head(5)

df_accented_dev.to_csv('cv-valid-dev-acc.csv') 





accent_true = (df2["accent"].notna())

df_accented_test = df2[accent_true]

print(df_accented_test.shape)

# df_accented_test.groupby("accent").count()

df_accented_test.to_csv('cv-valid-test-acc.csv')

df1.head(5)

df_accented_dev.head(5)