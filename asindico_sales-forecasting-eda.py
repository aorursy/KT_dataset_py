# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
print("we are dealing with ",len(train), "rows")
strain = train.sample(frac=0.01,replace=True)
items = pd.read_csv('../input/items.csv')
items.head()
print("we are dealing with",len(items['family'].unique()),"families of products")
f,axarr = plt.subplots(1,1,figsize=(15,10))

plt.xticks(rotation='vertical')

fhist = items.groupby(['family'],as_index=False).count()

sns.barplot(x=fhist['family'],y=fhist['class'])

stores = pd.read_csv('../input/stores.csv')

stores.head()
print("we are dealing with",len(stores['city'].unique()),'cities and',len(stores['state'].unique()),'states')
f,axar = plt.subplots(1,1,figsize=(15,10))

plt.xticks(rotation='vertical')

shist = stores.groupby(['city'],as_index=False).count()

sns.barplot(x=shist['city'],y=shist['state'])
trans = pd.read_csv('../input/transactions.csv')
trans.head()
df = strain.merge(right = items, on='item_nbr')
df = df.merge(right=stores,on='store_nbr')
df.head()
df['date']=pd.to_datetime(df['date'])
df['yday']=df['date'].dt.dayofyear
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
stores=[df['store_nbr'].unique()]
df.head()
grouped = df.groupby(['store_nbr','year','month','family']).sum().drop(['id','item_nbr','class','perishable','cluster','yday'],axis=1)
grouped.head(20)