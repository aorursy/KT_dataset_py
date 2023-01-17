import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split

import re

from nltk.stem.porter import *

plt.style.use('seaborn')

import plotly.express as px

from plotly import graph_objs as go

from collections import Counter

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from plotly.subplots import make_subplots

import plotly.graph_objects as go



pd.reset_option('^display.', silent=True)



# Load half the data and separate target from predictors

df = pd.read_csv('../input/hatred-on-twitter-during-metoo-movement/MeTooHate.csv', nrows=300000, encoding='latin1')



# Drop columns not used for modelling

cols_to_drop = ['status_id', 'created_at', 'location']

df.drop(cols_to_drop, axis=1, inplace=True)



# Convert text to string type

df['text'] = df['text'].astype(str)



# Rename category column to be more meaningful

df = df.rename(columns={"category": "hateful"})



print("Total number of samples:", len(df))



df.head()
# Print a random tweet as a sample

sample_index = 25

print(df.iloc[sample_index])
# Helper function to remove unwanted patterns

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt



# Remove Twitter handles from the data 

df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")



# Remove punctuations, numbers, and special characters

df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")



# Remove all words below 3 characters

df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))



# Tokenize the tweets

tokenized_tweet = df['text'].apply(lambda x: x.split())



# Stem the tweets

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])



# Put the processed tweets back in the dataframe

for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['text'] = tokenized_tweet
temp = df.hateful.value_counts()



fig = px.bar(temp)

fig.update_layout(

    title_text='Data distribution for each category',

    yaxis=dict(

        title='count'

    ),

    xaxis=dict(

        title='label'

    )

)

fig.show()
temp = df.groupby('hateful').count()['text'].reset_index()

temp['label'] = temp['hateful'].apply(lambda x : 'Hateful tweets' if x==1 else 'Non-hateful tweets')



fig = go.Figure(go.Funnelarea(

    text = temp.label,

    values = temp.text,

    title = {"position" : "top center", "text" : "Funnel Chart for target distribution"}

    ))

fig.show()
temp = df.groupby('hateful').count()['text'].reset_index()

fig = px.pie(temp, values='text', names=['Non-hateful', 'Hateful'],

             title="Pie chart of tweets")

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
tweet_len_non_hateful = df[df['hateful']==0]['text'].str.split().map(lambda x: len(x))

tweet_len_hateful = df[df['hateful']==1]['text'].str.split().map(lambda x: len(x))



fig = go.Figure()

fig.add_trace(go.Histogram(x=tweet_len_non_hateful, histfunc='avg', name="Non-hateful", opacity=0.75, histnorm='probability density'))

fig.add_trace(go.Histogram(x=tweet_len_hateful, histfunc='avg', name="Hateful", opacity=0.75, histnorm='probability density'))



fig.update_layout(

    title_text='Number of words in tweets', # title of plot

    xaxis_title_text='Value', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2,

    bargroupgap=0.1,

    barmode='overlay'

)

fig.show()
fig = px.scatter_matrix(df,

    dimensions=["favorite_count",

                "retweet_count",

                "followers_count",

                "friends_count",

                "statuses_count"],

    labels={col:col.replace('_', ' ') for col in df.columns}, # remove underscore

    color="hateful")



fig.update_layout(

    title='Scatter matrix of numerical variables',

    dragmode='select',

    width=800,

    height=800,

    hovermode='closest',

)

fig.show()
fig = px.scatter(df,

                 x="favorite_count",

                 y="retweet_count",

                 color="hateful",

                 labels={col:col.replace('_', ' ') for col in df.columns},

                 log_x=True,

                 log_y=True)



fig.update_layout(

    title='Favorite count vs retweet count',

)



fig.show()
fig = px.scatter(df,

                 x="favorite_count",

                 y="followers_count",

                 color="hateful",

                 labels={col:col.replace('_', ' ') for col in df.columns},

                 log_x=True,

                 log_y=True)



fig.update_layout(

    title='Favorite count vs followers count',

)



fig.show()
fig = px.scatter(df,

                 x="favorite_count",

                 y="statuses_count",

                 color="hateful",

                 labels={col:col.replace('_', ' ') for col in df.columns},

                 log_x=True,

                 log_y=True)



fig.update_layout(

    title='Favorite count vs statuses count',

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(histfunc="count", y=df['favorite_count'], x=df['hateful'], name="count", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="sum", y=df['favorite_count'], x=df['hateful'], name="sum", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="avg", y=df['favorite_count'], x=df['hateful'], name="avg", histnorm='probability'))



fig.update_layout(

    title_text='Count/sum/avg of favorite count', # title of plot

    xaxis_title_text='Hateful', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2,

    bargroupgap=0.1

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(histfunc="count", y=df['retweet_count'], x=df['hateful'], name="count", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="sum", y=df['retweet_count'], x=df['hateful'], name="sum", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="avg", y=df['retweet_count'], x=df['hateful'], name="avg", histnorm='probability'))



fig.update_layout(

    title_text='Count/sum/avg of retweet count', # title of plot

    xaxis_title_text='Hateful', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2,

    bargroupgap=0.1

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(histfunc="count", y=df['followers_count'], x=df['hateful'], name="count", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="sum", y=df['followers_count'], x=df['hateful'], name="sum", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="avg", y=df['followers_count'], x=df['hateful'], name="avg", histnorm='probability'))



fig.update_layout(

    title_text='Count/sum/avg of followers count', # title of plot

    xaxis_title_text='Hateful', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2,

    bargroupgap=0.1

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(histfunc="count", y=df['friends_count'], x=df['hateful'], name="count", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="sum", y=df['friends_count'], x=df['hateful'], name="sum", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="avg", y=df['friends_count'], x=df['hateful'], name="avg", histnorm='probability'))



fig.update_layout(

    title_text='Count/sum/avg of friends count', # title of plot

    xaxis_title_text='Hateful', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2,

    bargroupgap=0.1

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(histfunc="count", y=df['statuses_count'], x=df['hateful'], name="count", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="sum", y=df['statuses_count'], x=df['hateful'], name="sum", histnorm='probability'))

fig.add_trace(go.Histogram(histfunc="avg", y=df['statuses_count'], x=df['hateful'], name="avg", histnorm='probability'))



fig.update_layout(

    title_text='Count/sum/avg of statuses count', # title of plot

    xaxis_title_text='Hateful', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2,

    bargroupgap=0.1

)



fig.show()
df['temp_list'] = df['text'].apply(lambda x:str(x).split())



top = Counter([item for sublist in df['temp_list'].loc[df['hateful'] == 0] for item in sublist])

top_non_hateful = pd.DataFrame(top.most_common(25))

top_non_hateful.columns = ['Common_words','count']



fig = px.bar(top_non_hateful, x='count',y='Common_words',title='Common words in non-hateful tweets',orientation='h',width=700,height=700,color='Common_words')

fig.show()



fig = px.treemap(top_non_hateful, path=['Common_words'], values='count',title='Tree of common words in non-hateful twets')

fig.show()
top_non_hateful.columns = ['Common_words','count']

top_non_hateful.style.background_gradient(cmap='Purples')
fig = px.pie(top_non_hateful,

             values='count',

             names='Common_words',

             title='Word distribution in non-hateful tweets')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
top = Counter([item for sublist in df['temp_list'].loc[df['hateful'] == 1] for item in sublist])

top_hateful = pd.DataFrame(top.most_common(25))

top_hateful.columns = ['Common_words','count']

fig = px.bar(top_hateful, x='count',y='Common_words',title='Common words in hateful tweets',orientation='h',width=700,height=700,color='Common_words')

fig.show()



fig = px.treemap(top_hateful, path=['Common_words'], values='count',title='Tree of common words in hateful twets')

fig.show()
top_hateful.columns = ['Common_words','count']

top_hateful.style.background_gradient(cmap='Purples')
fig = px.pie(top_hateful,

             values='count',

             names='Common_words',

             title='Word distribution in hateful tweets')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
text = df['text'].loc[df['hateful'] == 0].values

cloud = WordCloud(stopwords=STOPWORDS,

                  background_color='white',

                  max_words=200,

                  max_font_size=100,

                  width=400,

                  height=200,

                  random_state=0).generate(str(text))



figure_size=(12,12)

plt.figure(figsize=figure_size)

plt.imshow(cloud, interpolation="bilinear");

plt.title("Word cloud of non-hateful tweets", fontdict={'size': 20, 'color': 'black', 

                           'verticalalignment': 'bottom'})

plt.axis('off')

plt.show()
text = df['text'].loc[df['hateful'] == 1].values

cloud = WordCloud(stopwords=STOPWORDS,

                  background_color='white',

                  max_words=200,

                  max_font_size=100,

                  width=400,

                  height=200,

                  random_state=0).generate(str(text))



figure_size=(12,12)

plt.figure(figsize=figure_size)

plt.imshow(cloud, interpolation="bilinear");

plt.title("Word cloud of hateful tweets", fontdict={'size': 20, 'color': 'black', 

                           'verticalalignment': 'bottom'})

plt.axis('off')

plt.show()