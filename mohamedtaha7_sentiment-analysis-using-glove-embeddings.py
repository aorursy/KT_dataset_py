# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O

import seaborn as sns # data visualization

import matplotlib.pyplot as plt # data visualization

from nltk.corpus import stopwords # text preprocessing - stopwords

import re, string, unicodedata # text preprocessing - regular expressions, string

from bs4 import BeautifulSoup # html processing

from wordcloud import WordCloud, STOPWORDS # visualizing word cloud from corpus & ignoring stopwords

from collections import Counter # counter for most common words

from sklearn.feature_extraction.text import CountVectorizer # feature-oriented counting of words

from sklearn.model_selection import train_test_split # splitting the dataset into train & test sets

from keras.preprocessing import text, sequence # word tokenization

from keras.models import Sequential # class to construct the model

from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Flatten # RNN layers to be used

from keras.optimizers import Adam # optimizer to be used

from keras.callbacks import ReduceLROnPlateau # learning rate decay on plateau
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

dataset.head(5)
dataset.info()
sns.set_style('darkgrid')

sns.countplot(dataset.sentiment)
stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()



#Removing the square brackets

def remove_between_square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)

# Removing URL's

def remove_between_square_brackets(text):

    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text

def remove_stopwords(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop and i.strip().lower().isalpha():

            final_text.append(i.strip().lower())

    return " ".join(final_text)

#Removing the noisy text

def denoise_text(text):

    text = strip_html(text)

    text = remove_between_square_brackets(text)

    text = remove_stopwords(text)

    return text

#Apply function on review column

dataset['review']=dataset['review'].apply(denoise_text)
plt.figure(figsize=(20,20))

cloud = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop).generate(" ".join(dataset[dataset.sentiment == 'positive'].review))

plt.grid(b=None)

plt.imshow(cloud, interpolation='bilinear')
plt.figure(figsize=(20,20))

cloud = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop).generate(" ".join(dataset[dataset.sentiment == 'negative'].review))

plt.grid(b=None)

plt.imshow(cloud, interpolation='bilinear')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))

text_len = dataset[dataset['sentiment'] == 'positive']['review'].str.len()

ax1.set_title('Positive Reviews')

ax1.hist(text_len, color='green')

text_len = dataset[dataset['sentiment'] == 'negative']['review'].str.len()

ax2.set_title('Negative Reviews')

ax2.hist(text_len, color='red')

fig.suptitle('Character Count in Reviews')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))

text_len = dataset[dataset['sentiment'] == 'positive']['review'].str.split().map(lambda x: len(x))

ax1.set_title('Positive Reviews')

ax1.hist(text_len, color='green')

text_len = dataset[dataset['sentiment'] == 'negative']['review'].str.split().map(lambda x: len(x))

ax2.set_title('Negative Reviews')

ax2.hist(text_len, color='red')

fig.suptitle('Word Count in Reviews')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))

text_len = dataset[dataset['sentiment'] == 'positive']['review'].str.split().apply(lambda x: [len(i) for i in x])

print("Test")

ax1.set_title('Positive Reviews')

sns.distplot(text_len.map(lambda x: np.mean(x)), ax=ax1, color='green')

text_len = dataset[dataset['sentiment'] == 'negative']['review'].str.split().apply(lambda x: [len(i) for i in x])

ax2.set_title('Negative Reviews')

sns.distplot(text_len.map(lambda x: np.mean(x)), ax=ax2, color='red')

fig.suptitle('Average Word Length in Reviews')
dataset['sentiment'] = pd.get_dummies(dataset['sentiment']).drop(['negative'], axis=1)
def get_corpus(texts):

    

    words = []

    

    for text in texts:

        for word in text.split():

            words.append(word.strip())

    

    return words



corpus = get_corpus(dataset.review)



corpus[:5]
print(len(corpus))
counter = Counter(corpus)

most_common = dict(counter.most_common(10))

most_common
def get_top_ngrams(corpus, n, g):

    

    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(16, 9))

most_common_uni = dict(get_top_ngrams(dataset.review, 20, 1))

sns.barplot(x=list(most_common_uni.values()), y=list(most_common_uni.keys()))
plt.figure(figsize=(16, 9))

most_common_bi = dict(get_top_ngrams(dataset.review, 20, 2))

sns.barplot(x=list(most_common_bi.values()), y=list(most_common_bi.keys()))
plt.figure(figsize=(16, 9))

most_common_uni = dict(get_top_ngrams(dataset.review, 20, 3))

sns.barplot(x=list(most_common_uni.values()), y=list(most_common_uni.keys()))
X_train, X_test, y_train, y_test = train_test_split(dataset.review, dataset.sentiment, train_size=0.9, random_state=0)

X_test_temp = X_test

y_test_temp = y_test
max_features = 10000

max_len = 128
tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)
tokenized_train = tokenizer.texts_to_sequences(X_train)

X_train = sequence.pad_sequences(tokenized_train, maxlen=max_len)



tokenized_test = tokenizer.texts_to_sequences(X_test)

X_test = sequence.pad_sequences(tokenized_test, maxlen=max_len) 
EMBEDDING_FILE = "../input/glove840b300dtxt/glove.840B.300d.txt"
def get_coeffs(word, *arr):

    return word, np.asarray(arr, dtype='float32')

embeddings_dict = dict(get_coeffs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_dict.values())

emb_mean, emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

num_words = min(max_features, len(word_index))



embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embed_size))



for word, i in word_index.items():

    

    if i >= num_words: continue

    embedding_vector = embeddings_dict.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
batch_size = 256

epochs=10

embed_size=300
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
model = Sequential()



model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=max_len, trainable=False))

model.add(Bidirectional(LSTM(units=128)))

model.add(Dropout(rate=0.8))

model.add(Dense(units=16, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))



model.compile(optimizer=Adam(lr=0.002), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[learning_rate_reduction])
print("Model Accuracy on Training Data: ", round(model.evaluate(X_train, y_train)[1]*100), "%")

print("Model Accuraccy on Testing Data: ", round(model.evaluate(X_test, y_test)[1]*100), "%")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

ax1.plot(history.history['accuracy'])

ax1.plot(history.history['val_accuracy'])

ax1.set_title('Training & Testing Accuracy')

ax1.set_xlabel('Epochs')

ax1.set_ylabel('Accuracy')

ax1.legend(['accuracy', 'val_accuracy'])

ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

ax2.set_title('Training & Testing Loss')

ax1.set_xlabel('Epochs')

ax1.set_ylabel('Loss')

ax2.legend(['loss', 'val_loss'])
