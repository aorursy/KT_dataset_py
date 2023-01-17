# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the dataset

df = pd.read_csv('../input/weatherHistory.csv')

df.head(3)
# Convert Formatted Date to DateTime

df['Date'] = pd.to_datetime(df['Formatted Date'])

df['year'] = df['Date'].dt.year

df['month'] = df['Date'].dt.month

df['day'] = df['Date'].dt.day

df['hour'] = df['Date'].dt.hour

df.info()
# Describe every month every year in dataset

from datetime import datetime, timedelta

import seaborn as sns

y = []

tempTemp = []

years = range(2006, 2017)

months = range(1, 4)

january25 = df.loc[(df['year'] == 2013) & (df['month'] == 1) & (df['day'] == 25)]['Date']

for day in range(1, 41):

    for hour in january25:

        temp = df.loc[(df['Date'] == (hour - np.timedelta64(day, 'D')))]['Temperature (C)']

    temp = temp.mean()

    y.append(temp)

y = list(reversed(y))

a = pd.DataFrame(np.array_split(y,5))

plt.figure(figsize=(10, 10))

sns.heatmap(a, annot=True)

plt.show()

y = []

for day in range(1, 41):

    for hour in january25:

        temp = df.loc[(df['Date'] == (hour + np.timedelta64(day, 'D')))]['Temperature (C)']

    temp = temp.mean()

    y.append(temp)

a = pd.DataFrame(np.array_split(y,5))

plt.figure(figsize=(10, 10))

sns.heatmap(a, annot=True)

plt.show()