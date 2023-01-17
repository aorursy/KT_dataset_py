!pip3 install ktrain
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import pandas as pd

import missingno as msno

import seaborn as sns

import plotly.graph_objects as go

#import plotly.express as px

import matplotlib.pyplot as plt

import spacy



from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from wordcloud import WordCloud, STOPWORDS 

import ktrain

from ktrain import text



from collections import Counter
df = pd.read_csv('../input/sentiment-analysis-for-financial-news/all-data.csv', encoding='cp437', header=None)

df.head()
df.columns = ["Sentiment", "News"]

df.head()
df.isnull().sum().any()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
nlp = spacy.load('en')



def normalise(msg):

    

    doc = nlp(msg)

    res = []

    

    for token in doc:

        if token.is_stop or token.is_punct or token.is_space or not(token.is_oov): #Removing Stop words and words out of vocabulary

            pass

        else:

            res.append(token.lemma_.lower())

            

    return res
df['News'] = df['News'].apply(normalise)

df.head()
fig = go.Figure([go.Bar(x=df.Sentiment.value_counts().index, y=df.Sentiment.value_counts().tolist())])

fig.update_layout(

    title="Values in each Sentiment",

    xaxis_title="Sentiment",

    yaxis_title="Values")

fig.show()
words_collection = Counter([item for sublist in df['News'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='YlGnBu', low=0, high=0, axis=0, subset=None)
word_string = " ".join(words_collection)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=1500, 

                      width=1000,

                      height=650

                         ).generate(word_string)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
pos_df = df[df['Sentiment'] == 'positive']

neg_df = df[df['Sentiment'] == 'negative']

neu_df = df[df['Sentiment'] == 'neutral']
words_collection = Counter([item for sublist in pos_df['News'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_string = " ".join(words_collection)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=1500, 

                      width=500,

                      height=650

                         ).generate(word_string)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
words_collection = Counter([item for sublist in neg_df['News'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_string = " ".join(words_collection)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=1500, 

                      width=500,

                      height=650

                         ).generate(word_string)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
words_collection = Counter([item for sublist in neg_df['News'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_string = " ".join(words_collection)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=1500, 

                      width=500,

                      height=650

                         ).generate(word_string)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
le = LabelEncoder()

df['Sentiment'] = le.fit_transform(df['Sentiment'])



df['News'] = df['News'].apply(lambda m: " ".join(m))



df.head()
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(df, 

                                                                    'News',

                                                                    label_columns=['Sentiment'],

                                                                   maxlen=500,

                                                                    preprocess_mode='bert')
model = text.text_classifier(name='bert',

                             train_data=(x_train, y_train),

                             preproc=preproc)
learner = ktrain.get_learner(model=model,

                             train_data=(x_train, y_train),

                             val_data=(x_test, y_test),

                             batch_size=6)
learner.fit_onecycle(lr=2e-5,

                     epochs=1)