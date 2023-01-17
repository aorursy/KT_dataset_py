import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

from operator import itemgetter

import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from wordcloud import WordCloud

import re



from tensorflow import keras

from keras.models import Model

from keras.layers import Dense, LSTM, Dropout, Embedding, Input, Bidirectional, GlobalMaxPooling1D, Conv1D, Activation, BatchNormalization

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load our data

data = pd.read_csv("../input/reviews-activities.csv", encoding="cp1252")
# preview the data

data.head()
data["Text"] = data["Text"].astype(str)
sns.countplot(y=data["Review-Activity"])
sns.countplot(y=data["Season"])
stop_words = set(stopwords.words('english'))

# use Tokenizer to get the top words for each season (removing stop words)

tokenize = Tokenizer()

tokenize.fit_on_texts(data[data.Season == "FALL"].Text)

fall_word_counts = sorted(tokenize.word_counts.items(), key=itemgetter(1), reverse=True)

fall_word_counts = [(word, count) for word, count in fall_word_counts if word not in stop_words]

tokenize.fit_on_texts(data[data.Season == "SUMMER"].Text)

summer_word_counts = sorted(tokenize.word_counts.items(), key=itemgetter(1), reverse=True)

summer_word_counts = [(word, count) for word, count in summer_word_counts if word not in stop_words]

tokenize.fit_on_texts(data[data.Season == "WINTER"].Text)

winter_word_counts = sorted(tokenize.word_counts.items(), key=itemgetter(1), reverse=True)

winter_word_counts = [(word, count)  for word, count in winter_word_counts if word not in stop_words]
# Review the top words by season

print("Fall Words\n", fall_word_counts[:15])

print("Summer Words\n", summer_word_counts[:15])

print("Winter Words\n", winter_word_counts[:15])
# fall word cloud

wordcloud = WordCloud()

wordcloud.generate_from_frequencies(frequencies=dict(fall_word_counts[:15]))

plt.figure(figsize=(10,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Convert text labels to integer values using Labelencoder and set the number of classes.

le_reviews = LabelEncoder()

le_reviews.fit(data["Review-Activity"].values)

num_review_classes = len(le_reviews.classes_)

data["Review-Activity_Label"] = le_reviews.transform(data["Review-Activity"].values)



le_season = LabelEncoder() 

le_season.fit(data["Season"].values)

num_season_classes = len(le_season.classes_)

data["Season_Label"] = le_season.transform(data["Season"].values)
# remove stop words and punctuation from text and stem words 

stemmer = PorterStemmer()

data["Text"] = data["Text"].apply(lambda x: " ".join([stemmer.stem(re.sub(r'[^\w\d]', "", word)) for word in x.lower().split() if word not in stop_words]))
# split data into features and labels

X = data["Text"]

Y = [data["Review-Activity_Label"].values, data["Season_Label"].values]
tokenize = Tokenizer()

tokenize.fit_on_texts(X)

# preview the word index

tokenize.word_index
# set maximum length of the sequences and the vocab size

max_length = 120

vocab_size = len(tokenize.word_index) + 1
# convert text (X) into integer sequences 

X = pad_sequences(tokenize.texts_to_sequences(X), maxlen=max_length, padding="post")
# set our input. Both branches will use the same input so we only need one.

input_1 = Input(shape=(max_length,))
# build our review-activities branch

reviews_activities_output = Embedding(vocab_size, 64)(input_1)

reviews_activities_output = Bidirectional(LSTM(16, return_sequences=True))(reviews_activities_output)

reviews_activities_output = Bidirectional(LSTM(8))(reviews_activities_output)

reviews_activities_output = Dense(16)(reviews_activities_output)

reviews_activities_output = BatchNormalization()(reviews_activities_output)

reviews_activities_output = Activation("relu")(reviews_activities_output)

reviews_activities_output = Dropout(rate=.4)(reviews_activities_output)

reviews_activities_output = Dense(num_review_classes, activation="softmax", name="reviews-activities-output")(reviews_activities_output)
# build season branch

season_output = Embedding(vocab_size, 64)(input_1)

season_output = Bidirectional(LSTM(16, return_sequences=True))(season_output)

season_output = Bidirectional(LSTM(8))(season_output)

season_output = Dense(16)(season_output)

season_output = BatchNormalization()(season_output)

season_output = Activation("relu")(season_output)

season_output = Dropout(rate=.4)(season_output)

season_output = Dense(num_season_classes, activation="softmax", name="season-output")(season_output)
model = Model(inputs=input_1, outputs=[reviews_activities_output, season_output])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(X, Y, epochs=5, validation_split=0.1)