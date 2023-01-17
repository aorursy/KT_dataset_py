import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import re



import matplotlib.pyplot as plt

import seaborn as sns



import plotly

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



import plotly.figure_factory as ff

import cufflinks as cf



%matplotlib inline

sns.set_style("whitegrid")

sns.set_context("paper")

plt.style.use('seaborn')
videos_df = pd.read_csv('../input/youtube-new/INvideos.csv')



category_df = pd.read_json('../input/youtube-new/IN_category_id.json')



display(videos_df.info())

display(category_df.info())
# Category data

display(category_df.sample(2))



display(category_df['items'][0])





# Video data

display(videos_df.sample(2))
# Fetching id and titles from the category dataset

list_id =[]

list_title = []



for index,row in category_df.iterrows():

    x = row['items']

    list_id.append(x['id'])

    list_title.append(x['snippet']['title'])

    

# Creating dataframe of id and title



category_df = pd.DataFrame(zip(list_id,list_title),columns=['category_id','category'])

category_df['category_id'] = category_df['category_id'].astype('int64')

category_df.head(2)
df = pd.merge(videos_df, category_df, on='category_id',how='inner')
category_details = df.groupby(['category']).agg({'video_id':'count','views':'sum','likes':'sum','dislikes':'sum','comment_count':'sum'}).reset_index()



fig =px.bar(data_frame=category_details.sort_values(by='video_id', ascending=False), 

           x='category', y='video_id',

           template='plotly_white')

fig.update_layout(

    title={

        'text': "Number of trending videos by category",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})
data = [go.Pie(

        labels = category_details['category'],

        values = category_details['video_id'],

        hoverinfo = 'label+value',

        title = 'Percentage of Videos'

    

)]



fig = plotly.offline.iplot(data, filename='category')
data = [go.Pie(

        labels = category_details['category'],

        values = category_details['views'],

        hoverinfo = 'label+value',

        title = '% of Views'

    

)]



plotly.offline.iplot(data, filename='category')
category_details.head(2)
# Views

views = category_details['views']

videos = category_details['video_id']



average_views = views/videos



# Likes

likes = category_details['likes']

average_likes = np.round(likes/average_views, 3)



# Dislikes

dislikes = category_details['dislikes']

average_dislikes = np.round(dislikes/average_views, 3)



# Comments

comments = category_details['comment_count']

average_comments = np.round(average_views/comments, 3)







category_details['average_views'] = average_views

category_details['average_likes'] = average_likes

category_details['average_dislikes'] = average_dislikes

category_details['average_comments'] = average_comments



category_details.head(2)
fig = plt.figure(figsize=(25,30))



ax1 = fig.add_subplot(411)

_ = sns.barplot(data=category_details, x='category', y='average_views', palette='Paired', ax=ax1)

xlabels = category_details['category'].to_list()

ylabels = category_details['average_views']

_ = ax1.set_title('Average Views per Video', fontsize=30)

_ = ax1.set_ylabel('Number of Views', fontsize=20)

_ = ax1.set_xlabel('')

_ = ax1.set_xticklabels(xlabels, rotation=30, fontsize=17)





ax2 = fig.add_subplot(412)

_ = sns.barplot(data=category_details, x='category', y='average_likes', palette='Paired', ax=ax2)

xlabels = category_details['category'].to_list()

ylabels = category_details['average_likes']

_ = ax2.set_title('Average Likes per Video', fontsize=30)

_ = ax2.set_ylabel('Number of Likes', fontsize=20)

_ = ax2.set_xlabel('')

_ = ax2.set_xticklabels(xlabels, rotation=30, fontsize=17)





ax3 = fig.add_subplot(413)

_ = sns.barplot(data=category_details, x='category', y='average_dislikes', palette='Paired', ax=ax3)

xlabels = category_details['category'].to_list()

ylabels = category_details['average_dislikes']

_ = ax3.set_title('Average Dislikes per Video', fontsize=30)

_ = ax3.set_ylabel('Number of Dislikes', fontsize=20)

_ = ax3.set_xlabel('')

_ = ax3.set_xticklabels(xlabels, rotation=30, fontsize=17)



ax4 = fig.add_subplot(414)

_ = sns.barplot(data=category_details, x='category', y='average_comments', palette='Paired', ax=ax4)

xlabels = category_details['category'].to_list()

ylabels = category_details['average_comments']

_ = ax4.set_title('Average Comments per Video', fontsize=30)

_ = ax4.set_ylabel('Number of Comments', fontsize=20)

_ = ax4.set_xlabel('')

_ = ax4.set_xticklabels(xlabels, rotation=30, fontsize=17)



fig.tight_layout(pad=0.5)
# Creating a df of num_of trending videos per channel 

trending_channels = df.groupby(['category','channel_title']).size().rename('num_videos').reset_index()



# Picking out the channel with the highest number of trending videos in each category

most_trending = trending_channels[trending_channels.groupby('category')['num_videos'].transform(max) == trending_channels['num_videos']]



most_trending
px.bar(data_frame=most_trending, 

           x='channel_title', y='num_videos',

           template='ggplot2',

           title='Channels with Highest no.of trending videos in each category')





fig = sns.barplot(data=most_trending, x='channel_title', y='num_videos', palette='Paired')

xlabels = (most_trending['category']+' / '+most_trending['channel_title']).to_list()

_ = plt.title('Channels with Highest no.of trending videos in each category', fontsize=20)

_ = plt.ylabel('Number of Videos', fontsize=20)

_ = plt.xlabel('')

_ = plt.xticks(np.arange(0,16),xlabels,rotation=270, fontsize=15)
import spacy

nlp = spacy.load('en_core_web_sm')



stop_words = spacy.lang.en.stop_words.STOP_WORDS



def preprocess_text(text):

    doc = nlp(text, disable=['ner','parser'])

    lemmas = [token.lemma_ for token in doc]

    a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stop_words]

    return ' '.join(a_lemmas)
categories_list = set(df['category'])

tags_category_dict = {}



# Create a dictionary of tags for each category

for category in categories_list:

    print(category)

    temp_tags_data = df[df['category'] == category]['tags']

    temp_tags_str = temp_tags_data.apply(preprocess_text)

    tags_category_dict[category] = temp_tags_str
tags_category_dict['Education'][:5]
from wordcloud import WordCloud

wc = WordCloud(background_color="black", stopwords=['song'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Music'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wc = WordCloud(background_color="black", stopwords=['song'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Gaming'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wc = WordCloud(background_color="black", stopwords=['comedy','video','funny'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Comedy'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wc = WordCloud(background_color="black", stopwords=['new','news','live'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['News & Politics'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()



wc = WordCloud(background_color="black", stopwords=['sports'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Sports'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wc = WordCloud(background_color="black", stopwords=['sports'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Science & Technology'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wc = WordCloud(background_color="black", stopwords=['sports'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Autos & Vehicles'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
wc = WordCloud(background_color="black", stopwords=['sports'],max_words=100, colormap="Set2")



string = ' '.join(tags_category_dict['Education'])

wc.generate(string.lower())

plt.figure(figsize=(13, 13))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()