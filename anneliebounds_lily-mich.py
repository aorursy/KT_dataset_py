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
df1 = pd.read_csv('/kaggle/input/michelin-restaurants/one-star-michelin-restaurants.csv')

df1.insert(1,'StarCount', 1, True)



df2 = pd.read_csv('/kaggle/input/michelin-restaurants/two-stars-michelin-restaurants.csv')

df2.insert(1,'StarCount', 2, True)



df3 = pd.read_csv('/kaggle/input/michelin-restaurants/three-stars-michelin-restaurants.csv')

df3.insert(1,'StarCount', 3, True)
df = pd.concat([df1, df2, df3], ignore_index=True)

row, column = df.shape

print('There are '+ str(row) + ' restaurants.')
df_cuisine = df.groupby(['cuisine'])['StarCount'].sum()

df_cuisine = df_cuisine.sort_values(ascending=False)

df_cuisine.head(1)
sweden_stars = df[df.region == 'Sweden']['StarCount'].sum()

print('Sweden has ' + str(sweden_stars) + ' Michelin restaurants.')
df_region = df.groupby('region')['StarCount'].count()

df_region.nlargest(3)
import matplotlib.pyplot as plt

import seaborn as sns



df_data = df.groupby(['region', 'cuisine']).apply(np.sum, 'StarCount')

df_plt = df_region.nlargest(10)

ax = sns.swarmplot(x='StarCount', y = 'region', hue = 'cuisine',

                   data = df_data, palette="Set2", dodge=True)



#df_plt = df.groupby('cuisine')['StarCount'].count()

#df_plt = df_plt.sort_values(ascending= False).head(5)

#df_plt = df_plt.plot(kind= 'barh')

plt.show()