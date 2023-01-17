# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
digimon = pd.read_csv('../input/digidb/DigiDB_digimonlist.csv')

move = pd.read_csv('../input/digidb/DigiDB_movelist.csv')

support = pd.read_csv('../input/digidb/DigiDB_supportlist.csv')
digimon.head()
move.head()
support.head()
digimon.info()
move.info()
support.info()
sns.pairplot(data=digimon[['Lv 50 HP', 'Lv50 SP', 'Lv50 Atk', 'Lv50 Def', 'Lv50 Int', 'Lv50 Spd']])

plt.show()
sns.countplot(x=digimon['Attribute'])

plt.show()
sns.countplot(x=digimon['Type'])

plt.show()
maxratio = move.copy()

maxratio['Max Ratio'] = move.Power / move['SP Cost']
maxratio['Max Ratio'].idxmax(axis=0)

print(move.iloc[80])
digimon.sort_values(by=['Lv50 Atk'], ascending=False).head(3)
digimon.sort_values(by=['Lv50 Def'], ascending=False).head(3)
sns.countplot(x=digimon['Type'])

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(hue=digimon['Type'], x=digimon['Stage'])

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(hue=digimon['Attribute'], x=digimon['Stage'])

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()