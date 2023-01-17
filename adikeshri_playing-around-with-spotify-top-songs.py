# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style='darkgrid')





import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1',index_col='Unnamed: 0')

data.sample(5)
plt.figure(figsize=(11,12))

sns.countplot(y=data['Artist.Name'],

              order=data['Artist.Name'].value_counts().index,

              palette='rocket')

plt.xlabel('No. of songs in spotify top 50 most listened song 2019')

plt.xticks([1,2,3,4])

plt.show()
plt.figure(figsize=(11,8))

sns.countplot(y=data['Genre'],

              order=data['Genre'].value_counts().index,

              palette='rocket')

plt.title('Most famous genres')

plt.show()
cols=data.columns

cols=cols.drop(['Track.Name','Artist.Name','Genre'])

cols
plt.figure(figsize=(20,20))

sns.set(style='whitegrid')

count=1

for col in cols:

    plt.subplot(4,3,count)

    count+=1

    c="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    sns.distplot(data[col],

                color=c)

    mean=np.mean(data[col])

    median=np.median(data[col])

    plt.scatter(mean,0,color='green',s=50,label='Mean')

    plt.scatter(median,0,color='maroon',s=50,label='Median')

    plt.legend(loc=0)

plt.tight_layout()

plt.show()
data.corr()
data.Genre.nunique()
from sklearn.cluster import KMeans

for k in range(2,22):

    modelKmeans=KMeans(k)

    X=data[cols]

    modelKmeans.fit(X)

    inertia=modelKmeans.inertia_

    print('Cost for k=' + str(k) +': ' + str(inertia))

labelsKmeans=modelKmeans.predict(X)

print(modelKmeans.get_params(deep=True))



from sklearn.mixture import GaussianMixture

for k in range(2,22):

    modelGmm=GaussianMixture(n_components=k)

    modelGmm.fit(X)

    print('Cost for k=' + str(k) + ': ' + str(modelGmm.score(X)))

labelsGmm=modelGmm.predict(X)

print(modelGmm.get_params(deep=True))
plt.figure(figsize=(12,40))

sns.set(style='whitegrid')

count=1

for col in cols:

    plt.subplot(10,2,count)

    plt.scatter(data.index,data[col],c=labelsKmeans,s=100)

    plt.title(col + ' (Clustered using KMeans)')

    count+=1

    plt.subplot(10,2,count)

    count+=1

    plt.scatter(data.index,data[col],c=labelsGmm,s=100)

    plt.title(col + ' (Clustered using Gaussian Mixture)')

plt.tight_layout()

plt.show()
labelsKmeans
labelsGmm