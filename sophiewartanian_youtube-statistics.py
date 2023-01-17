import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



CA = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv')

CA['country'] = "CA"

DE = pd.read_csv('/kaggle/input/youtube-new/DEvideos.csv')

DE['country'] = "DE"

FR = pd.read_csv('/kaggle/input/youtube-new/FRvideos.csv')

FR['country'] = "FR"

GB = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv')

GB['country'] = "GB"

US = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

US['country'] = "US"

IN = pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')

IN['country'] = "IN"



videos = pd.concat([CA, DE, FR, GB, US, IN])
# Top 10

trend = videos['channel_title'].value_counts()

trend = trend[:10,]



plt.figure(figsize=(15,7))

sns.set(font_scale=1)

sns.barplot(trend.index, trend.values, alpha=0.8)

plt.title('Top 10 Kanalen trending')

plt.ylabel('Aantal Videos Trending', fontsize=12)

plt.xlabel('Naam Kanaal', fontsize=12)

plt.xticks(rotation=45)

plt.show()
# Top 10 trending videos met aantal views, likes, dislikes en comments

top10 = videos.groupby('video_id', as_index=False).count().sort_values('title', ascending=False).head(10)

is_in_top10 = videos['video_id'].isin(top10['video_id'])

videos[is_in_top10].groupby('video_id', as_index=False).max()[['title', 'channel_title','views','likes','dislikes','comment_count']].sort_values(by=['views'], ascending=False)
DE = pd.read_csv('/kaggle/input/youtube-new/DEvideos.csv')

DE['country'] = "DE"



likes = DE['likes']

comments = DE['comment_count']

sns.scatterplot(likes, comments)

plt.xlabel("Aantal likes")

plt.ylabel("Aantal comments")

plt.title("Relatie Duitsland")

plt.show()
CA = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv')

CA['country'] = "CA"



likes = CA['likes']

comments = CA['comment_count']

sns.scatterplot(likes, comments)

plt.xlabel("Aantal likes")

plt.ylabel("Aantal comments")

plt.title("Relatie Canada")

plt.show()
US = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

US['country'] = "US"



likes = US['likes']

comments = US['comment_count']

sns.scatterplot(likes, comments)

plt.xlabel("Aantal likes")

plt.ylabel("Aantal comments")

plt.title("Relatie Verenigde Staten")

plt.show()
FR = pd.read_csv('/kaggle/input/youtube-new/FRvideos.csv')

FR['country'] = "FR"



likes = FR['likes']

comments = FR['comment_count']

sns.scatterplot(likes, comments)

plt.xlabel("Aantal likes")

plt.ylabel("Aantal comments")

plt.title("Relatie Frankrijk")

plt.show()
GB = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv')

GB['country'] = "GB"



likes = GB['likes']

comments = GB['comment_count']

sns.scatterplot(likes, comments)

plt.xlabel("Aantal likes")

plt.ylabel("Aantal comments")

plt.title("Relatie Groot-Britannië")

plt.show()
IN = pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')

IN['country'] = "IN"



likes = IN['likes']

comments = IN['comment_count']

sns.scatterplot(likes, comments)

plt.xlabel("Aantal likes")

plt.ylabel("Aantal comments")

plt.title("Relatie India")

plt.show()
gemiddeldAantalDagenTrendingPerLand = videos.groupby(['country', 'video_id']).count()[['title']].groupby('country').mean()

gemiddeldAantalDagenTrendingPerLand.columns = ['average number of trending days']

gemiddeldAantalDagenTrendingPerLand.plot(kind='bar', title='Average number of trending days per country')
videos['verschil_likes_dislikes'] = videos['dislikes'] - videos['likes']



top10_verschil_likes_dislikes = videos.groupby('video_id',as_index=False)['verschil_likes_dislikes'].max().sort_values(by=['verschil_likes_dislikes'], ascending=False).head(10)

is_in_top10_verschil_likes_dislikes = videos['video_id'].isin(top10_verschil_likes_dislikes['video_id'])

videos[is_in_top10_verschil_likes_dislikes].groupby('video_id', as_index=False).max()[['title', 'channel_title','views','likes','dislikes', 'verschil_likes_dislikes']].sort_values(by=['verschil_likes_dislikes'], ascending=False)



# Pak de tags column

tags = videos[['tags', 'views']]



# Split de eerste 5 tags en plaats in verschillende collumns

split_data = tags['tags'].str.split("|")

tags_df= pd.DataFrame(split_data.to_list())



# Hierdoor worden alleen de eerste 5 genoteerd

tags_df.drop(tags_df.iloc[:, 5:124], inplace = True, axis = 1)

tags_df.columns = ['first', 'second', 'third', 'fourth', 'fifth']



# Trim de quotes

columns = ['first', 'second', 'third', 'fourth', 'fifth']

tags_df[columns] = tags_df[columns].replace({'"':''}, regex=True)



# Sort per 1e tag

tags_df_sorted = tags_df.groupby('first', as_index=False).count().sort_values('second', ascending=False)[['first', 'second']]

tags_df_sorted.columns = ['tag', 'count']



# Plot

tags_df_sorted.head(10).plot(kind='bar',x='tag', figsize=(20,10))
# Data preparatie

tags_df= pd.DataFrame(split_data.to_list())



# Drop overbodige tags

tags_df.drop(tags_df.iloc[:, 1:124], inplace = True, axis = 1)

tags_df.columns = ['first_tag']



# Trim de quotes

columns = ['first_tag']

tags_df[columns] = tags_df[columns].replace({'"':''}, regex=True)



#Voeg views toe

tags_df.insert(1, column='views', value=tags['views'].to_list())



tags_df = tags_df.groupby(by=['first_tag'], axis=0).mean().sort_values('views', ascending = False)

tags_df.head(10).plot(kind='bar', figsize=(20,10))
videos['maand'] = videos['trending_date'].str[6:8]

videos.groupby('maand').sum()['views'].plot(figsize=(20,10), title='Sum of views per month', legend=True)