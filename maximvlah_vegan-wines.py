# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/barnivore-list-of-vegannonvegan-wines/barnivore_new.csv')



df.head()
df.info()
#Drop the null values

df = df.dropna()

df.info()
df.label.unique()
#Change the labels to vegan/not vegan

df.label = df.label.replace('NOT Vegan Friendly','Not Vegan')

df.label = df.label.replace('Not Vegan Friendly','Not Vegan')

df.label = df.label.replace('Vegan Friendly','Vegan')



#Delete the records with the 'Unknown' labels

df = df.loc[(df.label == 'Vegan') | (df.label == 'Not Vegan')]



df.label.unique()
df.info()
df.origin.unique()
#What are the top 10 countries with the highest number of user-labelled wines?

plt.figure(figsize = (16,8))

plt.title('Top 10 countries with the highest number of user-labelled wines')





# plt.pie()

sns.countplot(x = 'origin',

              hue = 'label',

              order = df['origin'].value_counts().iloc[:10].index,

              palette = 'rocket',

              data = df)

plt.xticks(rotation = 90)
#Who is the largest vegan-friendly producer in the US?

data = df.copy()



vegan = data.loc[(data.origin == 'USA') & (data.label == 'Vegan')]

non_vegan = data.loc[(data.origin == 'USA') & (data.label == 'Not Vegan')]



only_vegan = vegan.loc[vegan.producer.isin(non_vegan.producer).astype(int) == 0]



# #plot



plt.figure(figsize = (16,8))

plt.title('Top 10 uniquely vegan wine producers in the US')

sns.countplot(x = 'producer',

              order = only_vegan['producer'].value_counts().iloc[:10].index,

              palette = 'rocket',

              data = only_vegan)

plt.xticks(rotation = 90)