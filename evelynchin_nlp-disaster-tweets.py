import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')



from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict, Counter

from nltk.tokenize import word_tokenize



import re

import gensim

import string



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

from keras.initializers import Constant

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from tqdm import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# DATA PREPROCESSING



# 1. Load Data

# 2. Check Class Distribution

# 3. Compare Numbers of Characters and Words

# 4. Create Corpus Function

# 5. Identify Stopwords

# 6. Analyze Punctuation

# 7. Find Common Words

# 8. Ngram Analysis
# Load Data

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

train_copy = train.copy() # used when training model



print('Train Shape:', train.shape)

print('Test Shape:', test.shape)

train.head(3)
# Check Class Distribution

x = train.target.value_counts()

sns.barplot(x.index, x)

plt.gca().set_ylabel('Samples')
# Compare Numbers of Characters and Words



# Number of Characters

fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

tweet_len = train[train['target'] == 1]['text'].str.len()

ax1.hist(tweet_len, color='red')

ax1.set_title('Disaster Tweets')

tweet_len = train[train['target'] == 0]['text'].str.len()

ax2.hist(tweet_len, color='green')

ax2.set_title('Not Disaster Tweets')

fig.suptitle('Number of Characters in Tweets')

plt.show()



# Number of Words

fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

tweet_len = train[train['target'] == 1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len, color='red')

ax1.set_title('Disaster Tweets')

tweet_len = train[train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len, color='green')

ax2.set_title('Not Disaster Tweets')

fig.suptitle('Number of Words in Tweets')

plt.show()
# Create Corpus Function

def create_corpus(target):

    corpus = []

    for x in train[train['target'] == target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus



# Identify Stopwords

fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

stop = set(stopwords.words('english'))

corpus0 = create_corpus(0)

corpus1 = create_corpus(1)



# Class 0

dic = defaultdict(int)

for word in corpus0:

    if word in stop:

        dic[word] += 1

top = sorted(dic.items(), key = lambda x: x[1], reverse=True)[:10]

x, y = zip(*top)

ax1.bar(x, y, color='red')



# Class 1

dic = defaultdict(int)

for word in corpus1:

    if word in stop:

        dic[word]+=1

top = sorted(dic.items(), key = lambda x: x[1], reverse=True)[:10] 

x, y = zip(*top)

ax2.bar(x, y, color='green')
# Analyze Punctuation

fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

punchars = string.punctuation



# Class 0

dic = defaultdict(int)

for i in corpus0:

    if i in punchars:

        dic[i] += 1

x, y = zip(*dic.items())

ax1.bar(x, y, color='red')



# Class 1

dic = defaultdict(int)

for i in corpus1:

    if i in punchars:

        dic[i] += 1

x, y = zip(*dic.items())

ax2.bar(x, y, color='green')
# Find Common Words

counter = Counter(corpus0)

most = counter.most_common()

x = []

y = []

for word, count in most[:40]:

    if (word not in stop):

        x.append(word)

        y.append(count)

sns.barplot(x, y)
# Ngram Analysis



def get_top_bigrams(corpus, n = None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



plt.figure(figsize=(10,5))

top_bigrams = get_top_bigrams(train['text'])[:10]

x, y = map(list, zip(*top_bigrams))

sns.barplot(x, y)
# Data Cleaning

nTrain = train.shape[0]

df = pd.concat([train,test], axis=0, sort=False)



# Remove URLs

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)

df['text'] = df['text'].apply(lambda x: remove_URL(x))



# Remove HTML

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

df['text'] = df['text'].apply(lambda x: remove_html(x))



# Remove Emojis

def remove_emoji(text):

    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"

                               u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+", 

                               flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

df['text'] = df['text'].apply(lambda x: remove_emoji(x))



# Remove Punctuation

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)

df['text'] = df['text'].apply(lambda x : remove_punct(x))
# GloVe Vectorization



# 1. Define New Create Corpus Function

# 2. Set Up Embedding
# Define New Create Corpus Function

def create_corpus(df):

    corpus = []

    for tweet in tqdm(df['text']):

        words = [word.lower() for word in word_tokenize(tweet) if((word.isalpha() == 1) & (word not in stop))]

        corpus.append(words)

    return corpus



corpus = create_corpus(df)



# Set Up Embedding

embedding_dict = {}

with open('../input/glove-global-vectors-for-word-representation/glove.twitter.27B.100d.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word] = vectors

f.close()



max_len = 50

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences = tokenizer_obj.texts_to_sequences(corpus)

tweet_pad = pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')



word_index = tokenizer_obj.word_index

num_words = len(word_index) + 1

print('Number of unique words:', len(word_index))



embedding_matrix = np.zeros((num_words, 100))

for word, i in tqdm(word_index.items()):

    if i > num_words:

        continue

    emb_vec = embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec
# Model



# 1. Baseline Model

# 2. Plot Train and Validation

# 3. Submission
# Baseline Model

model = Sequential()

model.add(Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix), 

                    input_length = max_len, trainable=False))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

model.summary()
train = tweet_pad[:nTrain]

test = tweet_pad[nTrain:]



X_train, X_test, y_train, y_test = train_test_split(train, train_copy['target'], test_size=0.15)

print('Train:', X_train.shape)

print("Validation:", X_test.shape)



history = model.fit(X_train, y_train, batch_size=4, epochs=15, validation_data=(X_test, y_test), verbose=2)
# Plot Train and Validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training Loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training Accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation Accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Submission



sample_sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

y_pred = model.predict(test)

print(y_pred.shape)

y_pred = np.round(y_pred).astype(int).reshape(3263)

sub = pd.DataFrame({'id': sample_sub['id'].values.tolist(), 'target': y_pred})

sub.to_csv('submission.csv', index=False)