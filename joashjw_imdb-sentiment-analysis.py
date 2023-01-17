import pandas as pd

import tensorflow as tf

import numpy as np

import re

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
# reading data

data = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

data.head(10)
print(data.isnull().values.any()) # any NaN values

print(data.shape) # (row, column)

print(data['sentiment'].value_counts()) # check distribution of data
# change positive to 1, negative to 0

data['sentiment'] = data['sentiment'].replace({'positive':1, 'negative':0})

data['sentiment'] = data['sentiment'].astype(int)
# print stopwords

# A stop word is a commonly used word (such as “the”, “a”, “an”, “in”)

# Ignore these words as it takes up space and processing time

stop_words = set(stopwords.words('english'))

print(stop_words)



# Stemming words - convert to base/root word by reducing the word to their stems, faster method for good enough results

stemmer = SnowballStemmer("english")

print("\nStemming words: "+stemmer.stem('studies')+','+stemmer.stem('studying'))



# Lemmatization - convert to base/root word, considers part of speech as well, more complex and slower

lemmatizer = WordNetLemmatizer()

print("\nLemmatization: "+lemmatizer.lemmatize('studies')+','+lemmatizer.lemmatize('studying', pos='v'))
def clean_review(review):

    review = review.lower() # set to lower case

    review = re.sub("<.*?>|'s", '', review) # remove html tags and apostrophe

    # convert short forms back to their respective words

    review = re.sub("n't", ' not ', review) 

    review = re.sub("'ve", " have ", review)

    review = re.sub("can't", "cannot ", review)

    review = re.sub("i'm", "i am ", review)

    review = re.sub("'re", " are ", review)

    review = re.sub("'d", " would ", review)

    review = re.sub("'ll", " will ", review)

    # convert sentences to tokens - splitting them into their words

    tokens = word_tokenize(review)

    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()] # remove stopwords and non-alphanumeric words

    stemmed = [stemmer.stem(token) for token in filtered_tokens] # stemming the words

    return stemmed#" ".join(stemmed)



data['cleaned_review'] = data['review'].apply(clean_review)
data.head(10)
MAX_SEQUENCE_LENGTH = max(data['cleaned_review'].apply(len))

print(MAX_SEQUENCE_LENGTH)
# random_state for reproducibility, shuffle to shuffle data before splitting, stratify to maintain data distribution

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

x_train, x_test, y_train, y_test = train_test_split(data['cleaned_review'], data['sentiment'], test_size=0.3, random_state=10,

                                                    shuffle=True, stratify=data['sentiment'])



print(y_train.value_counts())

print(y_test.value_counts())
skip_gram = 1

min_count = 1

window = 3

word2vec = Word2Vec(x_train.values, window=window, min_count=min_count, sg=skip_gram, size=MAX_SEQUENCE_LENGTH)
# word2vec.wv.syn0 is the word2vec matrix

vocab_size, emdedding_size = word2vec.wv.syn0.shape

print("Vocab size: {}".format(vocab_size))



# this maps every word to a unique number

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, lower=True)

tokenizer.fit_on_texts(x_train)

train_sequence = tokenizer.texts_to_sequences(x_train)

# need to pad the sequences so that every row of data has the same number of input

train_sequence = tf.keras.preprocessing.sequence.pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH)



test_sequence = tokenizer.texts_to_sequences(x_test)

test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)



print("First row of input data: {}".format(train_sequence[0])) # padding of zeros added to match MAX_SEQUENCE_LENGTH

word_list = list(tokenizer.word_index.items())

print("Mapping of vocab to their respective number: {} ... {}".format(word_list[0], word_list[-1]))



# getting the word2vec weights for the embedding layer

# we have 1 to n number of vocab

# due to padding, zeros are added and have to cater for it too, 0 to n number of values

# expand the weights by 1 row filled with 0

embedding_weights = np.vstack([np.zeros((1,MAX_SEQUENCE_LENGTH)), word2vec.wv.syn0])
# input_dim is vocab+1

# set embedding layer to be not trainable as we have already trained it when building the word2vec

embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size+1, output_dim=emdedding_size, weights=[embedding_weights], trainable=False)

lstm_layer = tf.keras.layers.LSTM(128)

dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')



model = tf.keras.Sequential([embedding_layer, lstm_layer, dense_layer])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.summary()
history = model.fit(train_sequence, y_train, epochs=100, batch_size=32, steps_per_epoch=10)
probabilities = model.predict(test_sequence)

predictions = np.where(probabilities>0.5,1,0)

print("Accuracy: {}".format(accuracy_score(y_test, predictions)))

print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, predictions)))