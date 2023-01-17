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



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

import sys
df=pd.read_csv("../input/zomato.csv")

df.head()
df.shape
plt.figure(figsize=(10,7))

chains=df['name'].value_counts()[:30]

sns.barplot(x=chains,y=chains.index,palette='rocket')

plt.title("Most famous restaurants chains in Bangaluru")

plt.xlabel("Number of outlets")
plt.figure(figsize=(10,7))

locs=df['location'].value_counts()[:10]

sns.barplot(x=locs,y=locs.index,palette='deep')

plt.title("Top 20 spots in Bangalore with most number of Restaurants")

plt.xlabel("Number of outlets")




plt.rcParams['figure.figsize'] = (20, 9)

x = pd.crosstab(df['location'], df['online_order']=='Yes')

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['blue','pink'])

plt.title('location vs online order', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
df=df[df['dish_liked'].notnull()]

df.index=range(df.shape[0])

import re

likes=[]

for i in range(df.shape[0]):

    splited_array=re.split(',',df['dish_liked'][i])

    for item in splited_array:

        likes.append(item)



sns.barplot(pd.DataFrame(likes)[0].value_counts().head(10),pd.DataFrame(likes)[0].value_counts().head(10).index,orient='h')
df_yes=df[df['online_order']=='Yes']

df_no=df[df['online_order']=='No']

plt.figure(figsize=(10,7))

locs=df_yes['location'].value_counts()[:10]

sns.barplot(x=locs,y=locs.index,palette='rocket')

plt.title("Top 10 spots in Bangalore which takes online order")

plt.xlabel("NLocation")



pd.DataFrame(likes)[0].value_counts().tail(20)
rating_data=df[np.logical_and(df['rate'].notnull(), df['rate']!='NEW')]

rating_data.index=range(rating_data.shape[0])

import re

rating=[]

for i in range(rating_data.shape[0]):

    rating.append(rating_data['rate'][i][0:3])



rating_data['rate']=rating

rating_data.sort_values('rate',ascending=False)[['name','location','rate','rest_type']].head(60).drop_duplicates()
df_cdb=df[df['rest_type']=='Casual Dining, Bar']

df_cdb.head(5)



plt.figure(figsize=(15,10))

locs=df_cdb['location'].value_counts()[:30]

sns.barplot(x=locs,y=locs.index,palette='deep')

plt.title("Top 30 spots in Bangalore with most number of Restaurants of type 'Casual Dining, Bar'")

plt.xlabel("Number of outlets")


rating_data=df_cdb[np.logical_and(df_cdb['rate'].notnull(), df_cdb['rate']!='NEW')]

rating_data.index=range(rating_data.shape[0])

import re

rating=[]

for i in range(rating_data.shape[0]):

    rating.append(rating_data['rate'][i][0:3])



rating_data['rate']=rating

rating_data.sort_values('rate',ascending=False)[['name','location','rate','rest_type']].head(10).drop_duplicates()
plt.figure(figsize=(24, 18))



plt.subplot(2,1,1)

sns.countplot(x= 'rate', hue= 'online_order', data= df[df.rate != 0])

plt.title('Ratings vs online order', fontsize='xx-large')

plt.xlabel('Ratings', fontsize='large')

plt.ylabel('Count', fontsize='large')

plt.xticks(fontsize='large')

plt.xticks(fontsize='large')

plt.legend(fontsize='large')





df_cdb=df[df['rest_type']=='Quick Bites']

df_cdb.head(5)



plt.figure(figsize=(15,10))

locs=df_cdb['location'].value_counts()[:10]

sns.barplot(x=locs,y=locs.index,palette='rocket')

plt.title("Top 10 spots in Bangalore with most number of Restaurants of type 'Quick Bites'")

plt.xlabel("Number of outlets")