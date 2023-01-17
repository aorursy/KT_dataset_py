
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.listdir('../input/app-store-apple-data-set-10k-apps/')
description = pd.read_csv("../input/app-store-apple-data-set-10k-apps/appleStore_description.csv")
description.head()
df = pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv")
df.info()
df.describe()
df.drop(columns=['id','sup_devices.num','ipadSc_urls.num','lang.num','vpp_lic'],inplace=True)
df.drop(columns=['Unnamed: 0'],inplace=True)
df.head(10)
genre_group = df.groupby('prime_genre').mean()['user_rating']
genre_group
genre = [genre for genre,df in df.groupby('prime_genre')]

plt.bar(genre,genre_group)
plt.xticks(rotation='vertical')
plt.show()
free_apps = df.loc[df['price']==0.00]
paid_apps = df.loc[df['price']!=0.00]
free_apps_group = free_apps.groupby('prime_genre').mean()['user_rating']
paid_apps_group = paid_apps.groupby('prime_genre').mean()['user_rating']
plt.bar(genre,free_apps_group,label='free')
plt.plot(genre,paid_apps_group,color='red',label='paid')
plt.xticks(rotation='vertical')
plt.ylabel('Rating')
plt.xlabel('Genre')
plt.legend()
plt.show()
paid_apps.head()
paid_group = paid_apps.groupby('price').mean()['user_rating']
price = [price for price,df in paid_apps.groupby('price')]
plt.plot(price,paid_group,label='Paid apps rating')
plt.ylabel('Rating')
plt.xlabel('Price in $')
plt.grid()
plt.legend()
plt.show()
paid_group_price = paid_apps.groupby('prime_genre').mean()['price']
paid_group_price
plt.bar(genre,paid_group_price,label='Paid apps mean price')
plt.xticks(rotation='vertical')
plt.ylabel('Mean Price in $')
plt.xlabel('Genre')
plt.grid()
plt.legend()
plt.show()

df['Size in MB'] = round(df['size_bytes']/1000000)
df.drop(columns=['size_bytes'],inplace=True)
df = df[['track_name','Size in MB','currency', 'price', 'rating_count_tot',
       'rating_count_ver', 'user_rating', 'user_rating_ver', 'ver',
       'cont_rating', 'prime_genre']]
df.head()

size_group = df.groupby('Size in MB').mean()

size = [size for size,df in df.groupby('Size in MB')]
plt.plot(size,size_group['user_rating'],label='Rating according to size')
plt.xlabel('Size in MB')
plt.ylabel('Rating')
plt.legend()
plt.show()

plt.plot(size,size_group['price'],label='Price acc. to si')
plt.xlabel('Size in MB')
plt.ylabel('Price in $')
plt.show()
