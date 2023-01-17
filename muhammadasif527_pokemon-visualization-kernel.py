# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


Pokemon = pd.read_csv("../input/pokemon/PokemonData.csv")
Pokemon.head()
Pokemon.info
Pokemon.dtypes
Pokemon.describe()
Pokemon.describe(include='O')
Pokemon.describe(include='all')
Pokemon.isnull().sum()
sns.countplot(x='Legendary',hue='Legendary', data=Pokemon)
plt.figure(figsize=(15,6))

sns.countplot(x='Type1',data=Pokemon,order=Pokemon['Type1'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.countplot(x='Type2',data=Pokemon,order=Pokemon['Type2'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.countplot(x='Type1',hue='Legendary',data=Pokemon,order=Pokemon['Type1'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.countplot(x='Type2',hue='Legendary',data=Pokemon,order=Pokemon['Type2'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))

sns.distplot(Pokemon['HP'])
plt.figure(figsize=(15,6))

sns.kdeplot(Pokemon[Pokemon['Legendary'] == False]['HP'].dropna(),shade=True,label=False);

sns.kdeplot(Pokemon[Pokemon['Legendary'] == True]['HP'].dropna(),shade=True,label=True);
sns.set_context('poster')

plt.figure(figsize=(20,10))

sns.heatmap(Pokemon.corr(),annot=True,cmap='RdPu', fmt='1.3f');