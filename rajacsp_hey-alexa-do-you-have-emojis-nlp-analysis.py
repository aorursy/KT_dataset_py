# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install vaderSentiment

!pip install contextualSpellCheck
import random

import re



import plotly.express as px

import plotly.graph_objects as go



from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()



import spacy



#loading english module

nlp = spacy.load('en')



import warnings

warnings.filterwarnings('ignore')
FILEPATH = '/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv'
df = pd.read_csv(FILEPATH, delimiter = '\t')
df.sample(3)
df.describe()
df.info()
df.shape
df.isnull().sum()
df['variation'].sample(3)
import emoji



def get_emojis(content):

    return ''.join(c for c in content if c in emoji.UNICODE_EMOJI)



def get_emojis_count(content):

    content_gen = (c for c in content if c in emoji.UNICODE_EMOJI)

    return sum(1 for _ in content_gen)
content = '🤔 🙈 me así, bla es se 😌 ds 💕👭'
content
get_emojis(content)
df['emojis_count'] = df['verified_reviews'].apply(get_emojis_count)
df.sample(2)
df['emojis_count'].unique()
df['feedback'].unique()
df[df['emojis_count'] > 1]
df['variation'].unique()
# Which user collected the most?

emoji_df = df[df['emojis_count'] > 0]

emoji_df = pd.DataFrame(emoji_df['emojis_count'].value_counts().head(10)).reset_index()



emoji_df.sample(2)
state_fig = go.Figure(data=[go.Pie(labels=emoji_df['index'],

                             values=emoji_df['emojis_count'],

                             hole=.7,

                             title = 'Count by Emojis (more than zero)',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     ])

state_fig.update_layout(title = '% by Number of Emoji')

state_fig.show()
fig = px.scatter(df, x = "rating", y = "emojis_count",

                 color = "emojis_count", size='emojis_count', color_continuous_scale = 'Inferno')



fig.show()
# Get random review

def get_random_review(df, col = 'verified_reviews'):

    

    df = df.sample(n = 1)[col].item()

    

    return df
get_random_review(df)
# Get Sentiment Analysis by using Vader



def get_sentiment_score(sentence):

    

    score = analyser.polarity_scores(sentence)

    

    compound_score = int(float(score['compound']) * 10)

    

    return compound_score
get_sentiment_score(get_random_review(df))
df['sentiment_score'] = df['verified_reviews'].apply(get_sentiment_score)
df.sample(4)
def get_sentiment(content):

    

    score = analyser.polarity_scores(content)

    

    # {'neg': 0.0, 'neu': 0.326, 'pos': 0.674, 'compound': 0.7351}

    

    positive_score = (float(score['pos']) * 10)

    negative_score = (float(score['neg']) * 10)

    neutral_score = (float(score['neu']) * 10)

    compound_score = (float(score['compound']) * 10)

    

    return pd.Series([positive_score, negative_score, neutral_score, compound_score])
df[['sentiment_positive_score', 'sentiment_negative_score', 'sentiment_neutral_score', 'sentiment_compound_score']] = df['verified_reviews'].apply(get_sentiment)
df.sample(4)
# df['sentiment_score'].unique()
fig = px.scatter(df, x = "rating", y = "sentiment_compound_score",

                 color = "sentiment_compound_score", size='emojis_count', color_continuous_scale = 'Viridis')



fig.show()
fig = px.scatter(df, x = "sentiment_compound_score", y = "emojis_count",

                 color = "emojis_count", size='emojis_count', color_continuous_scale = 'inferno')



fig.show()
# import re



def get_words_count(content):

    """

        Get wordsc count excluding more than 2 letters

    """



    words = re.findall(r'\w{3,}', content.lower())

    

    return len(words)
df['words_count'] = df['verified_reviews'].apply(get_words_count)
df.sample(3)
fig = px.scatter(df, x = "sentiment_compound_score", y = "words_count",

                 color = "words_count", size='words_count', color_continuous_scale = 'inferno')



fig.show()
from spacy import displacy



def print_text_entities(text):

    

    print('review :')

    print(text)

    doc = nlp(text)

    for ent in doc.ents:

        print(f'\nEntity: {ent}, Label: {ent.label_}, {spacy.explain(ent.label_)}')

    

def show_ner():

    

    for i in range(5):

        df_item = df['verified_reviews'].sample(1)

        print('index: ', df_item.index.item())

        

        review_line = df_item.item()

        print_text_entities(review_line)

        doc = nlp(review_line)



        displacy.render(doc, style='ent', jupyter=True)



show_ner()