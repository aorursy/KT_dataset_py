import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]
sns.set_palette(palette = mycols, n_colors = 4)
print('My colours are set!')
df = pd.read_csv('../input/AppleStore.csv', index_col='Unnamed: 0')
print(f'Dataset has {df.shape[0]} observation, and {df.shape[1]} features')
df.head()
#Does dataset contain empty value ?
df.isnull().sum()
#That's good, no empty value in this dataset, next let's take a look into dataset information
df.describe()
df.describe(include='O')
#In currenc columns, there is just only one value. So we can safely remove it
df = df.drop('currency', axis='columns')
#I am gonna change size_byte = size_MB
df['size_Mb'] = df['size_bytes'].apply(lambda x: np.around(x / 1000000, decimals=2))
df.drop('size_bytes', axis='columns', inplace=True)

plt.subplots(figsize=(10,8))
bins = [0.00, 10.00, 20.00, 50.00, 100.00, 200.00, 500.00, 1000.00, 2000.00, np.inf]
labels = ['<10m', '10-20m', '20-50m', '50-100m', '100-200m', '200-500m', '500-1000m', '1G-2G', '>2G']
size = pd.cut(df['size_Mb'], bins, include_lowest=True, labels=labels)
sns.barplot(y=size.value_counts().values, x=size.value_counts().index)
indexs = np.argsort(df['price'])[::-1]
indexs = indexs[:30]
print(df.iloc[indexs][['track_name', 'price', ]])
bins = [-np.inf, 0.00, np.inf]
labels = ['free', 'not free']
df['price_categroies'] = pd.cut(df['price'], bins, include_lowest=True, labels=labels)

fig, axes=plt.subplots(figsize=(10,5))
price_df = df['price_categroies'].value_counts()
#axes.set_xticks([0,100,300, 500, 1000,5000])
sns.barplot(y=price_df.values, x=price_df.index)

free_app = df.loc[df['price_categroies'] == 'free']
free_app_rating = free_app.sort_values(by=['rating_count_tot'], ascending=False)

sns.barplot(x=rating_count_tot['rating_count_tot'][:10], y=rating_count_tot['track_name'][:10])
not_free_app = df.loc[df['price_categroies'] == 'not free']
not_free_app_rating = not_free_app.sort_values(by=['rating_count_tot'], ascending=False)

sns.barplot(x=not_free_app_rating['rating_count_tot'][:10], y=not_free_app_rating['track_name'][:10])
plt.subplots(figsize=(20,20))
rating_count_tot = df.sort_values(by=['rating_count_tot'], ascending=False)
sns.barplot(x=rating_count_tot['rating_count_tot'][:50], y=rating_count_tot['track_name'][:50])
rating_all = df['user_rating'].value_counts()
rating_all.sort_values(ascending=False,inplace=True)
plt.subplots(figsize=(15,10))
sns.barplot(x=rating_all.values, y= rating_all.index,order=rating_all.index, orient='h')
#lt.subplots(figsize=(10,8))
sns.barplot(y=df['cont_rating'].value_counts().values, x=df['cont_rating'].value_counts().index)
device_num = df['sup_devices.num'].value_counts()
device_num.sort_values(ascending=False,inplace=True)
plt.subplots(figsize=(10,8))
sns.barplot(x=device_num.values, y= device_num.index,order=device_num.index, orient='h')
screenshots = df['ipadSc_urls.num'].value_counts()
screenshots.sort_values(ascending=False,inplace=True)
plt.subplots(figsize=(10,8))
sns.barplot(x=screenshots.values, y= screenshots.index,order=screenshots.index, orient='h')
lang = df['lang.num'].value_counts()
lang.sort_values(ascending=False,inplace=True)
plt.subplots(figsize=(20,10))
sns.barplot(x=lang.values, y= lang.index,order=lang.index, orient='h')
genre = df['prime_genre'].value_counts()
genre.sort_values(ascending=False,inplace=True)
plt.subplots(figsize=(20,10))
sns.barplot(x=genre.values, y= genre.index,order=genre.index, orient='h')
game = df.loc[df['prime_genre'] == 'Games']
game.head()
price = (game['price'].value_counts()) / (game['price'].shape[0]) * 100
price.sort_values(ascending=False,inplace=True)

plt.subplots(figsize=(20,10))
ax = sns.barplot(y=price.values, x= price.index,order=price.index)
ax.set(xlabel='dollars', ylabel='% percent')
free_game = game.loc[game['price_categroies'] == 'free']
free_game_rating = free_game.sort_values(by=['rating_count_tot'], ascending=False)

not_free_game = game.loc[game['price_categroies'] == 'not free']
not_free_game_rating = not_free_game.sort_values(by=['rating_count_tot'], ascending=False)

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

#Most free and not free game apps with rating count of all versions
sns.barplot(x=free_game_rating['rating_count_tot'][:10], y=free_game_rating['track_name'][:10], ax=ax1)
sns.barplot(x=not_free_game_rating['rating_count_tot'][:10], y=not_free_game_rating['track_name'][:10], ax=ax2)
free_game_rating_cur = free_game.sort_values(by=['rating_count_ver'], ascending=False)
not_free_game_rating_cur = not_free_game.sort_values(by=['rating_count_ver'], ascending=False)

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

#Most free and not free game apps with rating count of current version  
sns.barplot(x=free_game_rating_cur['rating_count_ver'][:10], y=free_game_rating_cur['track_name'][:10], ax=ax1)
sns.barplot(x=not_free_game_rating_cur['rating_count_ver'][:10], y=not_free_game_rating_cur['track_name'][:10], ax=ax2)
ver1_game = game.loc[(game['ver'] == '1.0.0') & (game['rating_count_tot'] == game['rating_count_ver'])]
#Because there are still some app which version is less than 1.0.0, we need to search 
game.loc[(game['ver'] < '1') & (game['rating_count_tot'] <= game['rating_count_ver']) ]
#Ok, now we can say app version is 1.0.0 can be considered as new released 
new_game_rating = ver1_game.sort_values(by=['rating_count_tot'], ascending=False)

#Top 20 in new release
plt.subplots(figsize=(25,5))
sns.barplot(x=new_game_rating['rating_count_tot'][:20], y=new_game_rating['track_name'][:20])
