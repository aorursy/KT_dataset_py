# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns 
data = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
data
data.info()
null_columns=data.columns[data.isnull().any()]

data[null_columns].isnull().sum()


data[data['winPlacePerc'].isnull()]

data.drop(2744604, inplace=True)
sns.jointplot(x='winPlacePerc', y='kills', data=data, height=10, color="red")

plt.show()
sns.jointplot(x='winPlacePerc', y='damageDealt', data=data, height=10, color="maroon")

plt.show()
killcat=data.copy()

killcat['killCategory'] = pd.cut(killcat['kills'], [-1, 0, 5, 10, 20, 60], labels=['0_kills','1-5_kills', '6-10_kills', '11-20_kills', '10+_kills'])

plt.figure(figsize=(10,6))

sns.boxplot(x='killCategory', y='winPlacePerc', data=killcat)

plt.show()
sns.jointplot(x='winPlacePerc', y='walkDistance', data=data, height=10, color="green")

plt.show()
sns.jointplot(x='winPlacePerc', y='rideDistance', data=data, height=10,  color="blue")

plt.show()
data['healsandboosts'] = data['heals'] + data['boosts']

data[['heals', 'boosts', 'healsandboosts']].tail()
sns.jointplot(x='winPlacePerc', y='healsandboosts', data=data, height=10,  color="lime")

plt.show()
sns.jointplot(x='winPlacePerc', y='weaponsAcquired', data=data, height=10,  color="black")

plt.show()
sns.jointplot(x='winPlacePerc', y='revives', data=data, height=10,  color="#34495e")

plt.show()
revcat=data.copy()

revcat['revCategory'] = pd.cut(revcat['revives'], [-1, 0, 2, 5, 10, 40], labels=['0_revives','1-2_revives', '3-5_revives', '6-10_revives', '10+_revives'])

plt.figure(figsize=(10,6))

sns.boxplot(x='revCategory', y='winPlacePerc', data=revcat)

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='vehicleDestroys', y='winPlacePerc', data=data)

plt.show()