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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df
# Shows Na

df.isna().sum()
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.apply(lambda x: sum(x.isnull()), axis=0)

df['CabinNumber'] = df.Cabin.str[1:]

df['Cabins'] = df.Cabin.str[:1]

df['Cabins'] = df.Cabin.astype("category").cat.codes

df.Embarked = df.Embarked.astype("category").cat.codes

df = df.fillna(-1)

df
import matplotlib.pyplot as plt  

import seaborn as sns

correlation = df.corr()  

# Used to change the size of the figure 

plt.figure(figsize=(10, 10))  

sns.heatmap(correlation, vmax=1, annot=True, cmap='cubehelix')  
df.vertebrates.astype("category").cat.codes