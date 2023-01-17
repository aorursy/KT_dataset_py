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

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

df = pd.read_csv('/kaggle/input/toy-dataset/toy_dataset.csv')
df.head(5)

df.info()
df.Number.describe()
df.City.describe()

df.City.unique()
df.Income.describe()
df[df.Income < 0]
df.drop(df[df.Income < 0].index, inplace = True)
df.Income.describe()
df.boxplot(column ='Income', by ='City', figsize = (20,5))  #groupby('City')['Income'].

#plt.title('Mean income per City')
#df.groupby('Gender').Income.agg('mean')

df.boxplot(column = 'Income', by = 'Gender')
df.Age.describe()
def age_div(age):

    if age < 35:

        return 1

    if age < 45:

        return 2

    if age < 55:

        return 3

    else:

        return 4
df['Age_category'] = df.Age.apply(age_div)
df.head(3)
df.boxplot(column = 'Income', by = 'Age_category')
df.groupby('Age_category')['Income'].agg('mean').plot.bar()
df.Illness.value_counts().plot.barh()

df.boxplot(column = 'Income', by = 'Illness')
df.groupby('Illness').Income.agg('max').plot.bar()
df.Income.describe()
plt.figure(figsize=(16,7))

plt.xlabel('Income')

plt.ylabel('Freqency')

n, bins, patches = plt.hist(df[df.Gender == 'Male'].Income, bins = 150, color = 'gold', label = 'Male Income')

n, bins, patches = plt.hist(df[df.Gender == 'Female'].Income, bins = 150, color = 'crimson', label = 'Female Income')

plt.legend(loc='upper right')

plt.title('Income Frequency')
df.City.unique()
cities = ['Dallas', 'New York City', 'Los Angeles', 'Mountain View', 'Boston', 'Washington D.C.', 'Austin', 'San Diego']

colors = ['pink','indigo','blue','green','yellow','orange','brown','red']



plt.figure(figsize=(20,7))

for i,j in zip(cities,colors):

    n, bins, patches = plt.hist(df[df.City == i].Income, bins = 150, color = j, label = i)

plt.legend()

plt.xlabel('Income')

plt.ylabel('Frequency') 

plt.title('Income frequency per city')
df.groupby('City').agg('count')['Number'].plot.bar()

plt.title('Population per City')



pd.crosstab(df.City,df.Gender, normalize = 0).plot.bar(stacked = True)

plt.title('Gender dominance in working population per City')