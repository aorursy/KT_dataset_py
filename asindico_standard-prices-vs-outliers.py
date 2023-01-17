# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('../input/train.tsv',sep='\t')

df.head()
print("There are",len(df.brand_name.unique()),"brand names")
print("There are",len(df.category_name.unique()),"categories")
df.item_condition_id.unique()
import matplotlib

f,ax = plt.subplots(1,1,figsize=(15,20))

hist = df.groupby(['category_name'],as_index=False).count().sort_values(by='train_id',ascending=False)[0:25]

sns.barplot(y=hist['category_name'],x=hist['train_id'],orient='h')

matplotlib.rcParams.update({'font.size': 22})

plt.show()
hist['train_id'].values[0]/np.sum(hist['train_id'].values[1:])
import matplotlib.pyplot as plt

labels = hist['category_name'].values[0], hist['category_name'].values[1],hist['category_name'].values[2],hist['category_name'].values[3],'Others'

sizes = [hist['train_id'].values[0], hist['train_id'].values[1], hist['train_id'].values[2], hist['train_id'].values[3],np.sum(hist['train_id'].values[4:])]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','brown']

explode = (0.1, 0, 0, 0,0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
import matplotlib.pyplot as plt

labels =  'Top 20','Others'

sizes = [np.sum(hist['train_id'].values[0:20]),np.sum(hist['train_id'].values[20:])]

colors = ['gold', 'yellowgreen']

explode = (0.1, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
def nol(data, m=2):

    return data[abs(data - np.mean(data)) < m * np.std(data)]
def ol(data, m=3):

    return data[(data - np.mean(data)) >= m * np.std(data)]
k = 10

f,axarr = plt.subplots(5,2,figsize=(15,50))

matplotlib.rcParams.update({'font.size': 14})

for i in range(k):

    sns.distplot(nol(df[df['category_name']==hist['category_name'].values[i]].price),ax=axarr[int(i/2)][i%2])

    axarr[int(i/2)][i%2].set_title(hist['category_name'].values[i])


k = 10

f,axarr = plt.subplots(5,2,figsize=(15,50))

matplotlib.rcParams.update({'font.size': 14})

for i in range(k):

    sns.distplot(ol(df[df['category_name']==hist['category_name'].values[i]].price),ax=axarr[int(i/2)][i%2])

    axarr[int(i/2)][i%2].set_title(hist['category_name'].values[i])
from matplotlib_venn import venn2

k=10

f,axarr = plt.subplots(5,2,figsize=(15,30))

for i in range(k):

    obrand = set(df.iloc[(ol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].brand_name.unique())

    nobrand = set(df.iloc[(nol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].brand_name.unique())

    venn2(subsets = (len(obrand), len(nobrand),len(obrand.intersection(nobrand))),ax=axarr[int(i/2)][i%2],set_labels=('Price Outliers Brands','Standard Price Brands'))

    axarr[int(i/2)][i%2].set_title(df['category_name'][i])

    

    #print (obrand.intersection(nobrand))
from matplotlib_venn import venn2

k=10

f,axarr = plt.subplots(5,2,figsize=(15,35))

for i in range(k):

    obrand = set(df.iloc[(ol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].brand_name.unique())

    nobrand = set(df.iloc[(nol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].brand_name.unique())

    ohist = df.iloc[(ol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].groupby(['item_condition_id'],as_index=False).count()

    sns.barplot(x=ohist['item_condition_id'],y= ohist['train_id'],ax = axarr[int(i/2)][i%2])

    axarr[int(i/2)][i%2].set_title(df['category_name'][i])
from matplotlib_venn import venn2

k=10

f,axarr = plt.subplots(5,2,figsize=(15,35))

for i in range(k):

    obrand = set(df.iloc[(ol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].brand_name.unique())

    nobrand = set(df.iloc[(nol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].brand_name.unique())

    nohist = df.iloc[(nol(df[df['category_name']==hist['category_name'].values[i]].price).index).values].groupby(['item_condition_id'],as_index=False).count() 

    sns.barplot(x=nohist['item_condition_id'],y= nohist['train_id'],ax = axarr[int(i/2)][i%2])

    axarr[int(i/2)][i%2].set_title(df['category_name'][i])
import matplotlib

f,ax = plt.subplots(1,1,figsize=(15,20))

hist = df.groupby(['brand_name'],as_index=False).count().sort_values(by='train_id',ascending=False)[0:25]

sns.barplot(y=hist['brand_name'],x=hist['train_id'],orient='h')

matplotlib.rcParams.update({'font.size': 22})

plt.show()
df.groupby(['brand_name'],as_index=True).std().price.sort_values(ascending=False)[0:10]
df.groupby(['category_name'],as_index=True).std().price.sort_values(ascending=False)[0:10]