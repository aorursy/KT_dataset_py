# Import libraries and packages

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt   # visualizations

import plotly.express as px       # interactive visualizations

import seaborn as sns



# analyzing titles and descriptions

# Stopwords are words that don't add meaning to a sentence. Ex. the, them , it

from wordcloud import WordCloud, STOPWORDS                                                
df = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

df.head(3)
print(df.shape)

print(df.nunique())

df.dtypes



#checking for null values

print('\n')

df.info()
pd.options.display.float_format = '{:.2f}'.format #changes numbers from scientific notation to float

df.describe()
# Correlation matrix

plt.figure(figsize = (10,8))

corrMatrix = df[['views', 'likes', 'dislikes', 'comment_count']].corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
# create a new empty column to use category name instead of a number

df['category_name'] = np.nan



df.loc[(df['category_id'] == 1),'category_name'] = 'Film & Animation'

df.loc[(df['category_id'] == 2),'category_name'] = 'Cars & Vehicles'

df.loc[(df['category_id'] == 10),'category_name'] = 'Music'

df.loc[(df['category_id'] == 15),'category_name'] = 'Pets & Animals'

df.loc[(df['category_id'] == 17),'category_name'] = 'Sports'

df.loc[(df['category_id'] == 19),'category_name'] = 'Travel & Events'

df.loc[(df['category_id'] == 20),'category_name'] = 'Gaming'

df.loc[(df['category_id'] == 22),'category_name'] = 'People & Blogs'

df.loc[(df['category_id'] == 23),'category_name'] = 'Comedy'

df.loc[(df['category_id'] == 24),'category_name'] = 'Entertainment'

df.loc[(df['category_id'] == 25),'category_name'] = 'News & Politics'

df.loc[(df['category_id'] == 26),'category_name'] = 'How to & Style'

df.loc[(df['category_id'] == 27),'category_name'] = 'Education'

df.loc[(df['category_id'] == 28),'category_name'] = 'Science & Technology'

df.loc[(df['category_id'] == 29),'category_name'] = 'Non Profits & Activism'



df['category_name'].fillna('Not Categorized', inplace = True)
df.head(3)
#Check what kind of skew is with data

print(df[['views', 'likes', 'dislikes', 'comment_count']].skew())
#Histogram to see skew of views

fig, ax = plt.subplots() 

ax.hist(df['views'], bins = 30)

plt.show()
#Boxplot 

sns.boxplot(df['views'])
# create log transformations

df['likes_log'] = np.log(df['likes'] + 1)

df['views_log'] = np.log(df['views'] + 1)

df['dislikes_log'] = np.log(df['dislikes'] + 1)

df['comments_log'] = np.log(df['comment_count'] + 1)

df.describe()
fig, axs = plt.subplots(2, 2, figsize=(10,8))  



axs[0, 0].hist(df['views_log'], bins=25, color = 'blue', alpha=1)        # histogram 

axs[0, 0].set_ylabel('Count')

axs[0, 0].set_title('\nLog Views Histogram')

axs[0, 0].spines['right'].set_visible(False) # get rid of the line on the right

axs[0, 0].spines['top'].set_visible(False)   # get rid of the line on top



axs[0, 1].hist(df['comments_log'], bins=25, color = 'orange', alpha=1)        # histogram 

axs[0, 1].set_ylabel('Count')

axs[0, 1].set_title('\nLog Comment Count Histogram')

axs[0, 1].spines['right'].set_visible(False) # get rid of the line on the right

axs[0, 1].spines['top'].set_visible(False)   # get rid of the line on top



axs[1, 0].hist(df['likes_log'], bins=25, color = 'green', alpha=1)        # histogram 

axs[1, 0].set_ylabel('Count')

axs[1, 0].set_title('\nLog Likes Histogram')

axs[1, 0].spines['right'].set_visible(False) # get rid of the line on the right

axs[1, 0].spines['top'].set_visible(False)   # get rid of the line on top



axs[1, 1].hist(df['dislikes_log'], bins=25, color = 'red' , alpha=1)        # histogram 

axs[1, 1].set_ylabel('Count')

axs[1, 1].set_title('\nLog Dislikes Histogram')

axs[1, 1].spines['right'].set_visible(False) # get rid of the line on the right

axs[1, 1].spines['top'].set_visible(False)   # get rid of the line on top





fig.tight_layout(pad=.25)            #creates more space between figures

plt.show()
fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 26))



# Make orders for boxplots by median

views_order = df.groupby(by=["category_name"])["views_log"].median().sort_values(ascending = False).index

likes_order = df.groupby(by=["category_name"])["likes_log"].median().sort_values(ascending = False).index

dislikes_order = df.groupby(by=["category_name"])["dislikes_log"].median().sort_values(ascending = False).index

comments_order = df.groupby(by=["category_name"])["comments_log"].median().sort_values(ascending = False).index



# Log views Boxplot

sns.boxplot(x= 'category_name', y='views_log', data = df, ax = ax, palette = sns.color_palette('bright'),

            order = views_order)

ax.set_xticklabels(ax.get_xticklabels(),rotation=-40)

ax.set_title('Boxplot of Log Views by Category', fontsize=15)

ax.set_xlabel('')

ax.set_ylabel('Log Views', fontsize=15)



# log likes Boxplot

sns.boxplot(x= 'category_name', y='likes_log', data = df, ax = ax2, palette = sns.color_palette('bright'),

            order = likes_order)

ax2.set_xticklabels(ax2.get_xticklabels(),rotation=-40)

ax2.set_title('Boxplot of Log Likes by Category', fontsize=15)

ax2.set_xlabel('')

ax2.set_ylabel('Log Likes', fontsize=15)



# log dislikes Boxplot

sns.boxplot(x= 'category_name', y='dislikes_log', data = df, ax = ax3, palette = sns.color_palette('bright'),

            order = dislikes_order)

ax3.set_xticklabels(ax3.get_xticklabels(),rotation=-40)

ax3.set_title('Boxplot of Log Disikes by Category', fontsize=15)

ax3.set_xlabel('')

ax3.set_ylabel('Log Dislikes', fontsize=15)



# Log comments Boxplot

sns.boxplot(x= 'category_name', y='comments_log', data = df, ax = ax4, palette = sns.color_palette('bright'),

            order = comments_order)

ax4.set_xticklabels(ax4.get_xticklabels(),rotation=-40)

ax4.set_title('Boxplot of Log Comment Count by Category', fontsize=15)

ax4.set_xlabel('')

ax4.set_ylabel('Log Comment Count', fontsize=15)

sns.despine()



fig.tight_layout(pad=.5)

plt.show()
# Create new dataframe for the count of videos grouped by category name and then sorting by count

count_category = df.groupby('category_name').count().reset_index().sort_values(['video_id'], ascending=False)

count_category = count_category.rename(columns={'video_id': 'count'})



# bar chart

fig = px.bar(count_category, x='category_name', y = 'count', color = 'category_name')

fig.update_layout(showlegend=False, xaxis=dict(title_text=''))

fig.update_layout(title_text="Count of Trending Videos by Category", xaxis_tickangle= 30)



fig.show()
fig, ax = plt.subplots(figsize=(12,8))



# make variable to groupby channel and sort by the channels that most frequently have the trending videos

df_channel = df.groupby(['channel_title']).size().sort_values(ascending = False).head(20)



# plot

sns.barplot(df_channel.values, df_channel.index.values, palette = 'bright')

ax.set_title('The Most Frequently Trending Channels', fontsize = 25)

ax.set_xlabel('Count', fontsize = 16)

sns.despine()

ax.tick_params(axis = 'y', length = 0,labelsize = 15)



plt.show()
# remove ascii characters because Chinese characters are causing erros

df['title'] = df['title'].str.encode('ascii', 'ignore').str.decode('ascii')



# get n most frequent titles

n = 25



# create dataframe that only has videos that appear the most often (frequency)

longest_trending = df['video_id'].value_counts()[:n].index.tolist()

longest_trending

df_longest = df[df['video_id'].isin(longest_trending)]



# bar plot

fig, ax = plt.subplots(figsize=(10,8))

sns.countplot(y = 'title', data=df_longest, ax = ax, palette = 'bright', order = df_longest['title'].value_counts().index)

ax.set_ylabel('')

ax.set_xlabel('Count (Days)', fontsize = 18)

ax.set_title('The Longest Trending Videos by Days', fontsize = 25)

ax.tick_params(axis = 'y', length = 0,labelsize = 13)

sns.despine()

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



# Create the wordcloud object

wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_font_size=50, 

                      random_state=100).generate(str(df['title']))



# Display the generated image:

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("Most Frequent Words in Titles of Trending Videos", fontsize = 20)

plt.axis('off')

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)



# Create the wordcloud object

wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_font_size=50, 

                      random_state=100).generate(str(df['description']))



# Display the generated image:

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("Most Frequent Words in Descriptions of Trending Videos", fontsize = 20)

plt.axis('off')

plt.show()
fig, ax = plt.subplots(figsize=(10,5))



# make variable to count words in a column

title_count = [len(i.split()) for i in df['title'].tolist()]



# make plot

sns.kdeplot(title_count, shade=True)



ax.set_xlabel('Title Word Count')                     

ax.set_ylabel('Probability Density')

ax.set_title('Number of Words in Title Density Plot')

sns.despine()

plt.show()