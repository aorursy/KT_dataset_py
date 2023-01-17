# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
apple_store = pd.read_csv("../input/AppleStore.csv")
apple_store.head()
#print(1024*1024)
apple_store['MB']= apple_store.size_bytes.apply(lambda x:x/1024*1024)
#Most paid apps
outerlier = apple_store.sort_values(ascending=False,by='price')[['track_name','price']][:10]
apple_store.drop(outerlier.index,inplace=True)
print("MAXIMUM PRICE IN PLAY STORE:{}".format(apple_store.price.max()))
print("MININUM PRICE IN PLAY STORE:{}".format(apple_store.price.min()))
apple_store.price.hist(figsize=(14,7),log=True)
apple_store.prime_genre.value_counts()[:10]
yrange = [0,20]
fontsize = 17

plt.figure(figsize=(20,16))

plt.subplot(5,1,1)
games = apple_store[apple_store.prime_genre == 'Games']

plt.title('Games',fontsize=fontsize)
plt.ylim(yrange)
sns.stripplot(data=games,y='price',orient='h',jitter=True,color='#33aa99')
plt.xlabel('')

plt.subplot(5,1,2)
entertainment = apple_store[apple_store.prime_genre == 'Entertainment']
sns.stripplot(data=entertainment,y='price',jitter=True,orient='h',color='#aa7722')
plt.title('Entertainment',fontsize=fontsize)
plt.xlabel('')

plt.subplot(5,1,3)
education = apple_store[apple_store.prime_genre == 'Education']
sns.stripplot(data=education,y='price',jitter=True,orient='h',color='#66bb11')
plt.title('Education',fontsize=fontsize)
plt.xlabel('')

plt.subplot(5,1,4)
photo_video = apple_store[apple_store.prime_genre == 'Photo & Video']
sns.stripplot(data=photo_video,y='price',jitter=True,orient='h',color='#33ccaa')
plt.title('Photo & Video',fontsize=fontsize)
plt.xlabel('')

plt.subplot(5,1,5)
utilites = apple_store[apple_store.prime_genre == 'Utilities']
sns.stripplot(data=utilites,y='price',jitter=True,orient='h',color='#11bb77')
plt.title("Utilites",fontsize=fontsize)
plt.xlabel('')
plt.show()

print("There are {} category but for our analysis we choose only top 5 category".format(len(apple_store.prime_genre.unique())))
print(apple_store.prime_genre.value_counts()[:5])
#Compersion with free app with respective paid app category
top_five_category = apple_store[apple_store.prime_genre.isin(["Games",'Entertainment','Education','Photo & Video','Utilities'])]
free_app = top_five_category.prime_genre.value_counts()
paid_app = top_five_category.prime_genre.value_counts()

N = np.arange(5)
width = 0.60
plt.figure(figsize=(16,8))
p1 = plt.bar(N,free_app,width=width,color='#22bb00')
p2 = plt.bar(N,paid_app,width=width,bottom=free_app,color='#aabbcc')
plt.xticks(N,free_app.index.tolist())
plt.legend((p1[0],p2[0]),("Free","Paid"))

plt.figure(figsize=(15,7))
price = apple_store.groupby('price')['rating_count_tot'].agg('mean').reset_index()
plt.scatter(x=price.price,y=price.rating_count_tot)
plt.xlabel("Price in USD")
plt.ylabel("Average user rating count")
plt.title("Price of Apps Vs Average user rating count")
size_app = apple_store.groupby("MB")['rating_count_tot'].agg('mean').reset_index()
plt.figure(figsize=(15,7))
plt.scatter(x=size_app.MB,y=size_app.rating_count_tot)
plt.xlabel("Size of apps in MB")
plt.ylabel("Average user rating count")
plt.title("Size of apps Vs Average user rating count")
rating = apple_store.loc[:,['size_bytes','price','user_rating','user_rating_ver']]
rating.head()
rating.corr()
plt.figure(figsize=(15,7))
sns.heatmap(rating.corr(),cmap="YlGnBu",linewidths=0.5)
sns.boxplot(data=rating[['user_rating','user_rating_ver']])
plt.figure(figsize=(15,7))
sns.violinplot(x=top_five_category.prime_genre,y=top_five_category.user_rating_ver,inner=None)
#sns.swarmplot(x=top_five_category.prime_genre,y=top_five_category.user_rating_ver)
sns.stripplot(x='prime_genre',y='user_rating_ver',jitter=True,data=top_five_category,color='k',alpha=0.7)
#plt.ylim(0,None)
plt.xlabel("Category")
plt.ylabel("User Current Version Rating")
#sns.swarmplot(x='prime_genre',y='price',data=top_five_category)
