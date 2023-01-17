# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





import os

print(os.listdir("../input"))



data = pd.read_csv('../input/athlete_events.csv')

regions = pd.read_csv('../input/noc_regions.csv')

data.head(5)
data.describe()
data.info()
regions.head()
regions.describe()
data_merged = pd.merge(data, regions, on='NOC', how='left')

data_merged.head()
dataINA = data_merged[data_merged.NOC == 'INA']

dataINA.head()
dataINA.isnull().any()
dataINA = dataINA[dataINA.Medal.notnull()]

dataINA.head()
plt.figure(figsize=(20, 7))

plt.tight_layout()

sns.countplot(x='Year', hue='Medal', data= dataINA)

plt.title('Distribution of Indonesia Medals')
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(x='Sport', data= dataINA)

plt.title('Distribution of Indonesia Medals')
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(x='Year', hue='Sport', data= dataINA)

plt.title('Distribution of Medals by sports')
dataINA.Medal.value_counts()
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(x='Medal', hue='Year', data= dataINA)

plt.title('Distribution of Indonesia Medals by year')