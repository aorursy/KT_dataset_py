import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/covid19-coronavirus-romania/covid-19RO.csv', delimiter=',')

df.dataframeName = 'covid-19RO.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.tail(5)
df['date'] =  pd.to_datetime(df['date'])



df.info()
plt.plot(df['cases'])
plt.plot(df['recovered'])
plt.plot(df['deaths'])
plt.plot(df['tests'])