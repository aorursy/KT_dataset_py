import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

data=pd.read_csv("../input/zomato.csv")
data.head()
data.columns
print('We have nearly data of',data.shape[0],'food orders.' )
import seaborn as sns

sns.barplot(data.groupby('online_order').count().head()['url'].index,data.groupby('online_order').count().head()['url'])
plt.figure(figsize=(12,5))

sns.barplot(data['rest_type'].value_counts().head(8).index,data['rest_type'].value_counts().head(8))
print('SO This chart tells that most prefered hotel types are:' ,data['rest_type'].value_counts().head(8).index.values)
data=data[data['dish_liked'].notnull()]

data.index=range(data.shape[0])

import re

likes=[]

for i in range(data.shape[0]):

    splited_array=re.split(',',data['dish_liked'][i])

    for item in splited_array:

        likes.append(item)



sns.barplot(pd.DataFrame(likes)[0].value_counts().head(10),pd.DataFrame(likes)[0].value_counts().head(10).index,orient='h')
print("The above chart tells us that the Bangalorean people mostly prefer This foods:",pd.DataFrame(likes)[0].value_counts().head(10).index.values)
sns.barplot(pd.DataFrame(likes)[0].value_counts().tail(10),pd.DataFrame(likes)[0].value_counts().tail(10).index,orient='h')
print("The above chart tells us that the Bangalorean people hardly prefer these foods:",pd.DataFrame(likes)[0].value_counts().tail(10).index.values)
rating_data=data[np.logical_and(data['rate'].notnull(), data['rate']!='NEW')]

rating_data.index=range(rating_data.shape[0])

import re

rating=[]

for i in range(rating_data.shape[0]):

    rating.append(rating_data['rate'][i][:3])



rating_data['rate']=rating

rating_data.sort_values('rate',ascending=False)[['name','location','rate']].head(60).drop_duplicates()
print('This are the highest rated hotels in banglore:\n',rating_data.sort_values('rate',ascending=False)[['name']].head(60).drop_duplicates().values)
rating_data.sort_values('rate',ascending=True)[['name','location','rate']].head(50).drop_duplicates()
print('This are the highest rated hotels in banglore:\n',rating_data.sort_values('rate',ascending=True)[['name','location','rate']].head(50).drop_duplicates().values)