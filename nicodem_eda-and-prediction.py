# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib notebook
data = pd.read_csv("../input/80-cereals/cereal.csv")
#let us see the name of columns

data.columns
#let's see top 10 rows

data.head(20)
data.isnull().sum()
m = data.isin([-1]).any()

cols = m.index[m].tolist()

print(cols)
ads = data.isin([-1]).sum()

print(ads)
#lets see how many products are manufactured by each manufacturer

sns.countplot(data['mfr']).set(title='Counting of products manufactured by each company')

plt.xlabel('Manufacturer Name')

plt.show()
#lets see how many products are manufactured by each manufacturer categorised by type

sns.countplot(data['mfr'],hue=data['type']).set(title='Counting of products manufactured by each company')

plt.xlabel('Manufacturer Name')

plt.show()
# lets see the distribution of the ratings

plt.figure(figsize=(10,8))

sns.distplot(data['rating'],kde_kws={"color":'g'},hist_kws={"color":'r'},rug=True).set(title='Distribution of the ratings')

plt.show()
plt.figure(figsize=(12, 6))

a = sns.boxplot(data.mfr, data.rating)
plt.figure(figsize=(12, 6))

a = sns.violinplot(data.mfr, data.rating)
#joint plot of calories and rating 

sns.jointplot(x="calories", y="rating", kind='reg', data=data,color='r')
#joint plot of potass and rating

sns.jointplot(x="potass", y="rating", kind='reg', data=data,color='b')
#correlation plot in form of heatmap

plt.figure(figsize=(12, 12))

matrix = np.triu(data.corr())



sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1,square=True, center= 0,mask=matrix)
data.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)

plt.show()
palette = ['#e41a1c',]

pd.plotting.scatter_matrix(data,color=palette)

plt.show()