import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

path ='../input/weight-vs-age-of-chicks-on-different-diets/ChickWeight.csv'

data = pd.read_csv(path)
data.head()
data.shape
#create histogram for a dataset

import matplotlib.pyplot as plt

data.hist(figsize=(14,14), color='maroon', bins=20)

plt.show()
plt.figure(figsize=(12,6))

sns.scatterplot(x='Time',y='weight', hue="Diet", size='Diet', data=data)
plt.figure(figsize=(20,14))

sns.scatterplot(x='Time',y='weight', hue="Diet",data=data)
plt.figure(figsize=(10,7))

sns.swarmplot(x='Diet',y='Time',data=data)
df=data

df0 = df[df['Time'] == 0]

df2 = df[df['Time'] == 2]

df4 = df[df['Time'] == 4]

df6 = df[df['Time'] == 6]

df8 = df[df['Time'] == 8]

df10 = df[df['Time'] == 10]

df12 = df[df['Time'] == 12]

df14 = df[df['Time'] == 14]

df16 = df[df['Time'] == 16]

df18 = df[df['Time'] == 18]

df20 = df[df['Time'] == 20]

df21 = df[df['Time'] == 21]
df0.head()
df21.head()
#left figure is chicken's first weight, the right figure is after receiving 21 day of the diet

fig, ax =plt.subplots(1,2)

sns.swarmplot(x='Diet',y='weight',data=df0, ax=ax[0])

sns.swarmplot(x='Diet',y='weight',data=df21, ax=ax[1])

fig.show()