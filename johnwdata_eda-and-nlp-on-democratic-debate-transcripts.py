import numpy as np

import pandas as pd 



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Bidirectional, LSTM

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from sklearn.preprocessing import LabelEncoder, OneHotEncoder



import re

from operator import itemgetter

import collections



import nltk

from nltk.stem import PorterStemmer

from nltk.util import ngrams

from nltk import pos_tag

from nltk import RegexpParser

nltk.download('stopwords')



import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud



from collections import defaultdict



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/democratic-debate-transcripts-2020/debate_transcripts_v3_2020-02-26.csv", encoding="cp1252")
df.head()
df.shape
# Check for null values

df.isna().sum()
# Drop rows that do not contain a speaking time

df.dropna(inplace=True)
# Review all sections

df.debate_section.unique()
# Review all speakers

df.speaker.unique()
df.speaker.value_counts()
df_total_speaking_time = df.groupby(df.speaker)["speaking_time_seconds"].sum().sort_values()

# Review mean and median total speaking time

df_total_speaking_time.mean(), df_total_speaking_time.median()
# Let's drop speakers who had limited speaking time and view the remaining speakers

df = df[df.groupby(df.speaker)["speaking_time_seconds"].transform("sum") > 1100]

df.speaker.unique()
plt.figure(figsize=(20,7))

plt.xticks(fontsize=20)

plt.yticks(fontsize=15)

plt.ylabel('Total Speaking Time', fontsize=20)

plt.xlabel('Speaker', fontsize=20)

df.groupby(df.speaker)["speaking_time_seconds"].sum().sort_values().plot.bar()
# add a column for the speech with stop words and punctuation removed

stop_words = set(nltk.corpus.stopwords.words('english'))

for word in ["its", "would", "us", "then", "so", "it", "thats", "going", "also"]:

    stop_words.add(word)

df["speech_cleaned"] = df["speech"].apply(lambda x: " ".join([re.sub(r'[^\w\d]','', item.lower()) for item in x.split() if re.sub(r'[^\w\d]','', item.lower()) not in stop_words]))
# Let's look the most words used. 'People' has been by far the most used word.

t = Tokenizer()

t.fit_on_texts(df.speech_cleaned)

top_20_words = sorted(t.word_counts.items(), key=itemgetter(1), reverse=True)[:20]

top_20_words
# Create a word cloud with the most used words

wordcloud = WordCloud()

wordcloud.generate_from_frequencies(dict(top_20_words))

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Create a tokenizer for each candidate in order to create a bag of words for each.

tokenizers = defaultdict(Tokenizer)

for name in df.speaker.unique():

    tokenizers[name].fit_on_texts(df.speech_cleaned[df.speaker == name])
for candidate in df.speaker.unique():

    print(candidate , "\n", sorted(tokenizers[candidate].word_counts.items(), key=itemgetter(1), reverse = True)[:10], "\n")
fact_dict = dict()

for candidate in df.speaker.unique():

    if "fact" in tokenizers[candidate].word_index:

        fact_dict[candidate] = tokenizers[candidate].word_counts["fact"]

    else:

        fact_dict[candidate] = 0

sorted(fact_dict.items(), key=itemgetter(1), reverse=True)
# Healthcare appears to be a common word as well, let's see which candidates speak use the word 'healthcare' most often

healthcare_dict = dict()

for candidate in df.speaker.unique():

    if "healthcare" in tokenizers[candidate].word_index:

        healthcare_dict[candidate] = tokenizers[candidate].word_counts["healthcare"]

    else:

        healthcare_dict[candidate] = 0

sorted(healthcare_dict.items(), key=itemgetter(1), reverse=True)
# Tokenize all text in order

text = ""

tokenized = list()

for speech in df.speech_cleaned:

    text += " " + speech

tokenized = text.split()
# Most common bi-grams

n_grams = collections.Counter(ngrams(tokenized, 2))

n_grams.most_common(10)
# Most common tri-grams

n_grams = collections.Counter(ngrams(tokenized, 3))

n_grams.most_common(10)
# Most common quad-grams

n_grams = collections.Counter(ngrams(tokenized, 4))

n_grams.most_common(10)
# Most common quint-grams

n_grams = collections.Counter(ngrams(tokenized, 5))

n_grams.most_common(10)
# Tokenize all Joe Biden text in order

text = ""

tokenized = list()

for speech in df.speech_cleaned[df.speaker=="Joe Biden"]:

    text += " " + speech

tokenized = text.split()
# Most common Biden bi-grams

n_grams = collections.Counter(ngrams(tokenized, 2))

n_grams.most_common(10)
# Most common Biden tri-grams

n_grams = collections.Counter(ngrams(tokenized, 3))

n_grams.most_common(10)
# Tokenize all Bernie Sanders text in order

text = ""

tokenized = list()

for speech in df.speech_cleaned[df.speaker=="Bernie Sanders"]:

    text += " " + speech

tokenized = text.split()
# Most common Bernie Sanders bi-grams

n_grams = collections.Counter(ngrams(tokenized, 2))

n_grams.most_common(10)
# Most common Bernie Sanders tri-grams

n_grams = collections.Counter(ngrams(tokenized, 3))

n_grams.most_common(10)
# review the speech lengths

df_speech_length = df["speech"].apply(lambda x: len(x.split()))

df_speech_length.hist(bins=30)

df_speech_length.mean(), df_speech_length.median(), np.percentile(df_speech_length, 80)
# set max sequence length

max_len = 150
# Load reviews for sentiment analysis

df_reviews = pd.read_csv("../input/imdb-reviews/dataset.csv", encoding="cp1252")
df_reviews.head()
# Create stemmer

stemmer = PorterStemmer()
# Clean up review text

df_reviews["SentimentTextCleaned"] = df_reviews["SentimentText"].apply(lambda x: " ".join([stemmer.stem(re.sub(r'[^\w\d]','', item.lower())) for item in x.split() if re.sub(r'[^\w\d]','', item.lower()) not in stop_words]))                                                                                                      
# Take a look at cleaned up reviews

df_reviews["SentimentTextCleaned"][:10]
review_tokenize = Tokenizer()

review_tokenize.fit_on_texts(df_reviews["SentimentTextCleaned"])

X_sentiment = pad_sequences(review_tokenize.texts_to_sequences(df_reviews["SentimentTextCleaned"]), maxlen=max_len, padding="post")

Y_sentiment = df_reviews["Sentiment"]
# build a model for sentiment analysis

sentiment_model = Sequential([

    Embedding(len(review_tokenize.word_index) + 1, 64),

    Bidirectional(LSTM(32, return_sequences=True)),

    Bidirectional(LSTM(16)),

    Dense(64),

    BatchNormalization(),

    Activation("relu"),

    Dropout(.25),

    Dense(16),

    BatchNormalization(),

    Activation("relu"),

    Dropout(.25),

    Dense(2, activation="softmax")

])
sentiment_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
sentiment_model.fit(X_sentiment, Y_sentiment, validation_split=.1, epochs=4)
# Stem cleaned up speech in the debate data

df["speech_cleaned"] = df["speech_cleaned"].apply(lambda x: " ".join([stemmer.stem(item) for item in x.split()]))

# Review stemmed speech

df["speech_cleaned"][:10]
# Add a column to show the sentiment of each speech

predictions = []

for speech in df["speech_cleaned"]:

  prediction = sentiment_model.predict(pad_sequences(review_tokenize.texts_to_sequences([speech]), maxlen=max_len, padding="post"))

  predictions.append(prediction[0][1])

df["sentiment"] = predictions
# Review sentiment by speaker. The higher the number the more positive their speech is.

plt.figure(figsize=(20,7))

plt.xticks(fontsize=20)

plt.yticks(fontsize=15)

plt.ylabel("Average Sentiment", fontsize=20)

plt.xlabel("Speaker", fontsize=20)

df.groupby(df.speaker)["sentiment"].mean().sort_values().plot.bar()