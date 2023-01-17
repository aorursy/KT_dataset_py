# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

import time

data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.head()
data.info()
y = data['diagnosis'] 

unnecessary_columns = ['id','Unnamed: 32','diagnosis']

x = data.drop(unnecessary_columns,axis = 1)
x.head()
ax = sns.countplot(y,label="count")

M, B = y.value_counts()

print("Malignant count", M)

print("Benign count", B)
x.describe()
data = x

data_std = (data - data.mean()) / data.std()
data_std.describe()
data_1 = pd.concat([y, data_std.iloc[:, 0:10]], axis=1)

data_1 = pd.melt(data_1, id_vars='diagnosis',var_name='features',value_name='values')

plt.figure(figsize=(10,10))

sns.violinplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_1, split = True,inner = 'quart' )

plt.xticks(rotation=45)
data_2 = pd.concat([y, data_std.iloc[:, 10:20]], axis=1)

data_2 = pd.melt(data_2, id_vars='diagnosis',var_name='features',value_name='values')

plt.figure(figsize=(10,10))

sns.violinplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_2, split = True,inner = 'quart' )

plt.xticks(rotation=45)
data_3 = pd.concat([y, data_std.iloc[:, 20:30]], axis=1)

data_3 = pd.melt(data_3, id_vars='diagnosis',var_name='features',value_name='values')

plt.figure(figsize=(10,10))

sns.violinplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_3, split = True,inner = 'quart' )

plt.xticks(rotation=45)
plt.figure(figsize=(15,10))

sns.boxplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_1)

plt.xticks(rotation=45)
plt.figure(figsize=(15,10))

sns.boxplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_2)

plt.xticks(rotation=45)
plt.figure(figsize=(15,10))

sns.boxplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_3)

plt.xticks(rotation=45)
sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind = 'regg')
sns.set(style='whitegrid',palette='muted')

plt.figure(figsize=(10,10))

sns.swarmplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_1)

plt.xticks(rotation=45)
sns.set(style='whitegrid',palette='muted')

plt.figure(figsize=(10,10))

sns.swarmplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_2)

plt.xticks(rotation=45)
sns.set(style='whitegrid',palette='muted')

plt.figure(figsize=(10,10))

sns.swarmplot(x = 'features', y= 'values', hue = 'diagnosis',data= data_3)

plt.xticks(rotation=45)
f, ax = plt.subplots(figsize = (18,18))

sns.heatmap(x.corr(),annot = True, linewidths=.5, fmt = '.1f',ax=ax)