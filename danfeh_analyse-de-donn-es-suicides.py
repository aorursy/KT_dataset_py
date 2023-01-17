# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/master.csv")
data.head()
data.describe()
data.info()
data.shape
data.columns
data.corr()
plt.figure(figsize=(12, 12))

p = sns.heatmap(data.corr(), annot=True)

data.country.unique()
p = sns.countplot(data.sex)

plt.xticks(rotation = 90)

plt.show()
data_by_country_meam = data.groupby('country').mean()

mean_suicide = data_by_country_meam[["suicides/100k pop"]]

mean_suicide.sort_values('suicides/100k pop', ascending = False)[:10]
mean_suicide.sort_values('suicides/100k pop')[:10]
data.isna().sum()
pl = plt.figure(figsize=(20, 20))

data.groupby('country').suicides_no.count().plot('barh')

plt.xlabel('Numéro total de suicides', fontsize=14)

plt.ylabel('Pays', fontsize= 21)

plt.title('Suicide par pays', fontsize= 19)

plt.show()
pl = plt.figure(figsize=(20, 20))

data.groupby(['generation']).generation.count().plot('bar')