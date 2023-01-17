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
import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

input = pd.read_csv('../input/zomato.csv')
input.shape
input.head(1)
banglore=input[['name','location','rate','approx_cost(for two people)','rest_type','listed_in(type)']]

banglore['name'].nunique()
banglore.columns
banglore.duplicated().sum()

banglore.drop_duplicates(inplace=True)


banglore.rate.replace(('NEW','-'),np.nan,inplace =True)

# first make it as string

banglore.rate = banglore.rate.astype('str')

# remove the "/5" 

banglore.rate = banglore.rate.apply(lambda x: x.replace('/5','').strip())

# convert column type to float

banglore.rate = banglore.rate.astype('float')

banglore.isna().sum()
banglore.name.apply(lambda x: x.title())

banglore.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(type)': 'meal_type'}, inplace=True)

banglore.head()

# we will do the same for rate column

bins =[0,2,3,4,5]

labels =['low','moderate','high','exceptional']

banglore['rate_range'] = pd.cut(banglore.rate, bins=bins,labels=labels)

banglore['rate_range'].head()
banglore.head()
banglore.meal_type.unique()
for i in banglore.columns:

    print(i,"column has",banglore[i].isnull().sum(),"nulls")

#filling null values in dataframe by 0

banglore.dropna(inplace=True)
for i in banglore.columns:

    print(i,"column has",banglore[i].isnull().sum(),"nulls")
banglore.columns
banglore.drop('rest_type',axis=1,inplace=True)



types_dummy=pd.get_dummies(banglore['meal_type'],prefix='type')



types_dummy.head()
banglore=pd.concat([banglore,types_dummy],axis=1)



banglore.drop('meal_type',axis=1,inplace=True)



banglore.drop_duplicates(inplace=True)



banglore.reset_index(inplace=True)



banglore.drop('index',axis=1,inplace=True)



banglore.head()
from matplotlib.pyplot import figure

figure(figsize=(12,4))

plt.scatter(banglore['name'].head(9),banglore['rate'].head(9))



# plotting parameters

# set the color for all graphs

colors = ['pink' for i in range(banglore.location.nunique())]

colors[0] = 'blue'

# histogram for restaurants average_rate

plt.rcParams['figure.figsize'] = 13,6

plt.subplot(1,2,1)

banglore.rate.hist(color='green')

plt.axvline(x= banglore.rate.mean(),ls='--',color='orange')

plt.title('Rate Distribution',weight='bold')

plt.xlabel('Rate')

plt.ylabel('Count')



plt.subplot(1,2,2)

banglore.rate_range.value_counts().plot('bar',color=colors,grid=True)

plt.title('Rate range Distribution',weight='bold')

plt.xlabel('Rate range')

plt.ylabel('Number of restaurants')

plt.xticks(rotation=0)

plt.tight_layout();
banglore.columns
banglore.average_cost = banglore.average_cost.apply(lambda x: int(x.replace(',','')))

# check for values

banglore.average_cost.unique()


banglore_clus=banglore.drop(['name','location','rate_range'],1)



#creating duplicates for standardization purpose and comparing results

banglore_clus_std=banglore_clus

banglore_clus_copy=banglore_clus



banglore_clus.head()
from sklearn.cluster import KMeans

#1 without standardizing input features

kmeans=KMeans(n_clusters=3,random_state=0)

kmeans.fit(banglore_clus)


labels=kmeans.labels_



banglore_clus['clusters'] = labels



banglore_clus.head()
clmns=banglore.columns.values.tolist()



clmns.extend(['clusters'])



clmns
import seaborn as sns

sns.lmplot('average_cost', 'rate', data=banglore_clus, fit_reg=False, hue="clusters",  scatter_kws={"marker": "D", "s": 100})

plt.title('Clusters average_Cost_level vs rating')

plt.ylabel('rating')

plt.xlabel('price_level')