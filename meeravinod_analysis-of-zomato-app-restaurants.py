# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import math

from statistics import mode

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')
df.head()
df.info()
df.nunique()
df['listed_in(type)'].unique()
df['dish_liked'].head()[0]
df['reviews_list'][0]
df.menu_item.head()
df.isnull().sum()
msno.matrix(df)
(df['dish_liked'].isnull().sum())/(df.shape[0])
df.drop(['url','phone','menu_item'],axis=1,inplace=True)
df.columns



df.rename(columns={'listed_in(type)':'rest_category', 'listed_in(city)':'city','rate':'rating','approx_cost(for two people)':'cost_for_two'},inplace= True)
df.head()


df.drop_duplicates(keep='last',inplace=True)



df.duplicated().sum()
df.rating.unique()

df.rating.replace('-',np.nan, inplace=True)

df.rating.replace('NEW',np.nan,inplace=True)

string_float = lambda x: x.replace('/5', '') if type(x) == np.str else x

df.rating = df.rating.apply(string_float).str.strip().astype('float')

df.rating.head()

       
df.rating

type(df.rating[0])
df['cost_for_two'] = df['cost_for_two'].astype(str)

df['cost_for_two']=df['cost_for_two'].apply(lambda x: x.replace(',','.'))

df['cost_for_two']=df['cost_for_two'].astype(float)
for i in df['cost_for_two'].unique():

    if i<10.0:

        df['cost_for_two'].replace({i:i*1000}, inplace=True)

      

        
df['cost_for_two'].unique()
list=[]

for i in df['dish_liked']:

    if type(i)!=float:

        i.split(',')

        list.append(i)

# print(list)

print(mode(list))
# 10 most liked dishes in bangalore

from collections import Counter

words_to_count = (word for word in list if word[:1].isupper())

c = Counter(words_to_count)

print (c.most_common(10))
df.describe()
#most voted

df[df['votes']==df['votes'].max()]
#Most expensive

df[df['cost_for_two']==df['cost_for_two'].max()]
#most affordable restaurant

df[df['cost_for_two']==df['cost_for_two'].min()]
cuisine_list=[]

for i in df['cuisines']:

    if type(i)!=float:

        i.split(',')

        cuisine_list.append(i)

# print(cuisine_list)

new_list=[]

for i in cuisine_list:

    new_list.append(i.split(','))

new_new_list = []

for i in new_list:

    for j in i:

        new_new_list.append(j.strip())

# print(new_new_list)



from collections import Counter

words_to_count = (word for word in new_new_list if word[:1].isupper())

c = Counter(words_to_count)

print (c.most_common(10))
sns.distplot(df.cost_for_two,kde=False,bins=50)
plt.figure()

sns.countplot(df.online_order)
sns.countplot(df.book_table)
df.rating.isnull().sum()/df.shape[0]
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

sns.distplot( df.rating, ax=axes[0])

sns.boxplot( df.rating, ax=axes[1])
fig=plt.figure(figsize=(10,3))

sns.countplot(df.rest_category)

plt.tight_layout()
c= df.city.value_counts()[:10]

c.plot.bar()
s= df.rest_type.value_counts()[:10]

s.plot.bar()
bt_plot=pd.crosstab(df['rating'], df['book_table'])

bt_plot.plot(kind='bar',stacked=True);
popular=df['name'].value_counts()[:10]

# popular.plot.bar()

sns.barplot(x=popular,y=popular.index,palette='Set1')
# most popular outlets

popular=df['name'].value_counts()[:20]

print(popular)