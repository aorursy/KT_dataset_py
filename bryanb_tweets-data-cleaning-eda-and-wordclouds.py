# Libraries

import re

import numpy as np

import pandas as pd

import plotly.graph_objects as go

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
# Import data

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')



# Adding interesting variables for analysis

train['length'] = train['text'].apply(lambda x:len(str(x)))

train['word_counts'] = train['text'].apply(lambda x:len(str(x).split()))
# Print first rows of train

train.head()

print(f'{train.shape[0]} observations, {train.shape[1]} columns')
# Count samples per category

print(train['sentiment'].value_counts(), "\n")

print(train['sentiment'].value_counts(normalize=True))
x = ['Neutral', 'Positive', 'Negative']

y = [11118, 8582, 7781]



# Use the hovertext kw argument for hover text

fig = go.Figure(data=[go.Bar(x=x, y=y,

            hovertext=['40% of tweets', '31% of tweets', '29% of tweets'])])



# Customize aspect

#marker_color='rgb(158,202,225)'

fig.update_traces(marker_line_color='midnightblue',

                  marker_line_width=1.)

fig.update_layout(title_text='Distribution of sentiment')

fig.show()
neutral = train[train['sentiment'] == 'neutral']

positive = train[train['sentiment'] == 'positive']

negative = train[train['sentiment'] == 'negative']
#neutral_text

print("Neutral tweet example  :",neutral['text'].values[1])

# Positive tweet

print("Positive Tweet example :",positive['text'].values[1])

#negative_text

print("Negative Tweet example :",negative['text'].values[1])
x = train.length.values



fig = go.Figure(data=[go.Histogram(x=x,

                                   marker_line_width=1, 

                                   marker_line_color="midnightblue", 

                                   xbins_size = 5)])



fig.update_layout(title_text='Distribution of tweet lengths')

fig.show()
x1 = neutral.length.values

x2 = positive.length.values

x3 = negative.length.values



fig = go.Figure(data=[go.Histogram(x=x1,

                                   marker_line_width=1, 

                                   marker_line_color="midnightblue", 

                                   xbins_size = 5, 

                                   opacity = 0.5)])



fig.update_layout(title_text='Distribution of neutral tweet lengths')

fig.show()



fig = go.Figure(data=[go.Histogram(x=x2,

                                   marker_line_width=1, 

                                   marker_color='rgb(50,202,50)', 

                                   xbins_size = 5, 

                                   marker_line_color="midnightblue", 

                                   opacity = 0.5)])



fig.update_layout(title_text='Distribution of positive tweet lengths')

fig.show()



fig = go.Figure(data=[go.Histogram(x=x3,

                                   marker_line_width=1, 

                                   marker_color='crimson', 

                                   xbins_size = 5, 

                                   marker_line_color="midnightblue", 

                                   opacity = 0.5)])



fig.update_layout(title_text='Distribution of negative tweet lengths')

fig.show()
y1 = neutral.length.values

y2 = positive.length.values

y3 = negative.length.values



fig = go.Figure()



fig.add_trace(go.Box(y=y1, 

                     name="Neutral", 

                     marker_line_width=1, 

                     marker_line_color="midnightblue"))



fig.add_trace(go.Box(y=y2, 

                     name="Positive", 

                     marker_line_width=1, 

                     marker_color = 'rgb(50,202,50)'))



fig.add_trace(go.Box(y=y3, 

                     name="Negative", 

                     marker_line_width=1, 

                     marker_color = 'crimson'))



fig.update_layout(title_text="Box Plot tweet lengths")



fig.show()
indexes = [index for index, tweet in enumerate(train['text']) if len(str(tweet)) <= 5 ]

train.iloc[indexes,:]
indexes = [index for index, tweet in enumerate(train['text']) if len(str(tweet)) == 3 ]

train.iloc[indexes,:]
x = train.word_counts.values



fig = go.Figure(data=[go.Histogram(x=x,

                                   marker_line_width=1, 

                                   marker_line_color="midnightblue")])



fig.update_layout(title_text='Distribution of tweet lengths')

fig.show()
x1 = neutral.word_counts.values

x2 = positive.word_counts.values

x3 = negative.word_counts.values



fig = go.Figure(data=[go.Histogram(x=x1,

                                   marker_line_width=1, 

                                   marker_line_color="midnightblue", 

                                   opacity = 0.5)])



fig.update_layout(title_text='Distribution of neutral tweet lengths')

fig.show()



fig = go.Figure(data=[go.Histogram(x=x2,

                                   marker_line_width=1, 

                                   marker_color='rgb(50,202,50)', 

                                   marker_line_color="midnightblue", 

                                   opacity = 0.5)])



fig.update_layout(title_text='Distribution of positive tweet lengths')

fig.show()



fig = go.Figure(data=[go.Histogram(x=x3,

                                   marker_line_width=1, 

                                   marker_color='crimson', 

                                   marker_line_color="midnightblue", 

                                   opacity = 0.5)])



fig.update_layout(title_text='Distribution of negative tweet lengths')

fig.show()
y1 = neutral.word_counts.values

y2 = positive.word_counts.values

y3 = negative.word_counts.values



fig = go.Figure()



fig.add_trace(go.Box(y=y1, 

                     name="Neutral", 

                     marker_line_width=1, 

                     marker_line_color="midnightblue"))



fig.add_trace(go.Box(y=y2, 

                     name="Positive", 

                     marker_line_width=1, 

                     marker_color = 'rgb(50,202,50)'))



fig.add_trace(go.Box(y=y3, 

                     name="Negative", 

                     marker_line_width=1, 

                     marker_color = 'crimson'))



fig.update_layout(title_text="Box Plot word counts")



fig.show()
train.info()
sentences = train['selected_text'].values

print(list(sentences).index(np.nan))
print(sentences[13133])

train = train.drop(13133, axis=0)
# Find emoji patterns

emoji_pattern = re.compile("["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           "]+", flags=re.UNICODE)
# Basic function to clean the text

def clean_text(text):

    text = str(text)

    # Remove emojis

    text = emoji_pattern.sub(r'', text)

    # Remove identifications

    text = re.sub(r'@\w+', '', text)

    # Remove links

    text = re.sub(r'http.?://[^/s]+[/s]?', '', text)

    return text.strip().lower()



train['text'] = train['text'].apply(lambda x:clean_text(x))
def wordcloud(df, text = 'text'):

    

    # Join all tweets in one string

    corpus = " ".join(str(review) for review in df[text])

    print (f"There are {len(corpus)} words in the combination of all review.")

    

    wordcloud = WordCloud(max_font_size=50, 

                          max_words=100,

                          collocations = False,

                          background_color="white").generate(corpus)

    

    plt.figure(figsize=(15,15))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()



wordcloud(df = train)
print('Neutral Wordcloud')

wordcloud(df = neutral)



print('Positive Wordcloud')

wordcloud(df = positive)



print('Negative Wordcloud')

wordcloud(df = negative)