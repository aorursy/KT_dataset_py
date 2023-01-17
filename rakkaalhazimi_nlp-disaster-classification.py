# Data Science Purpose

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set_style('darkgrid')

from sklearn.feature_extraction.text import CountVectorizer



# Natural Language Processing Purpose

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

from nltk.util import ngrams

from nltk.tokenize import word_tokenize

import gensim



# Python built-in modules

import re

import string

import statistics

from tqdm import tqdm

from collections import defaultdict, Counter



# Deep learning purpose

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read csv

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

target = train['target']



# Data shape

print("Train shape", train.shape)

print("Test shape", test.shape)
cdist = train['target'].value_counts()

sns.barplot(cdist.index, cdist.values)

plt.gca().set_title("Class Distribution");
# Missing data on both training and test data

train_miss = train.isnull().sum()

test_miss = test.isnull().sum()

print("Missing values in train data", train_miss, sep="\n", end="\n\n")

print("Missing values in test data", test_miss, sep="\n")



# Iterate throught tuples of 3 values (name, missing data, real data)

for name, data, real in [('Train', train_miss, train), ('Test', test_miss, test)]:

    fig, ax = plt.subplots()

    ax.set_title(f"Missing Values on {name} data", pad=20)

    

    values = data.values / len(real) # Divide values with len of real data to get the percentage

    index = data.index # Index names

    

    bar = sns.barplot(x=data.index, y=data.values / len(real), ax=ax)

    ax.set_ylabel("Percentage")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))



first_len = train[train['target'] == 0]['text'].str.len()

second_len = train[train['target'] == 1]['text'].str.len()



ax1.hist(first_len, color='g')

ax1.set_ylabel('Characters length')

ax1.set_title('Not Disaster Tweets')



ax2.hist(second_len, color='r')

ax2.set_title('Disaster Tweets')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))



first_words = train[train['target'] == 0]['text'].str.split().map(lambda x: len(x))

second_words = train[train['target'] == 1]['text'].str.split().map(lambda x: len(x))



fig.suptitle("Number of words in tweets")



ax1.hist(first_words, color='g')

ax1.set_ylabel('Words length')

ax1.set_title('Not Disaster Tweets')



ax2.hist(second_words, color='r')

ax2.set_title('Disaster Tweets')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))





avg_one = train[train['target'] == 0]['text'].str.split().map(lambda x: [len(i) for i in x])

avg_two = train[train['target'] == 1]['text'].str.split().map(lambda x: [len(i) for i in x])



fig.suptitle("Average of words in tweets")



sns.distplot(avg_one.apply(lambda x: statistics.mean(x)), color='g', ax=ax1)

ax1.set_ylabel('Words Average')

ax1.set_title('Not Disaster Tweets')



sns.distplot(avg_two.apply(lambda x: statistics.mean(x)), color='r', ax=ax2)

ax2.set_title('Disaster Tweets')
def create_corpus(target):

    corpus = []

    

    for x in train[train['target'] == target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
for i in [0, 1]:

    fig, ax = plt.subplots()

    corpus = create_corpus(i)



    dic = defaultdict(int)

    for word in corpus:

        if word in stop:

            dic[word] += 1



    top = sorted(dic.items(), key=lambda x:x[1], reverse=True)[:10]

    x, y = zip(*top)



    ax.bar(x, y, color='brown' if i == 0 else 'orange')

    ax.set_title(f'Class {i}');
for i in [0, 1]:

    fig, ax = plt.subplots(figsize=(10, 5))

    corpus = create_corpus(i)

    

    dic = defaultdict(int)

    special = string.punctuation

    for word in corpus:

        if word in special:

            dic[word] += 1



    x, y = zip(*dic.items())

    ax.bar(x, y, color='brown' if i == 0 else 'orange')

    ax.set_title(f'Class {i}')
corpus = create_corpus(0)

counter = Counter(corpus)

most = counter.most_common()



x, y = [], []



for word, count in most[:40]:

    if word not in stop:

        x.append(word)

        y.append(count)

fig, ax = plt.subplots(figsize=(10, 5))

fig.suptitle('Common Words')



sns.barplot(x=y, y=x, orientation='horizontal', ax=ax);
def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10, 5))

top_tweet_bigram = get_top_tweet_bigrams(train['text'])[:10]

x, y = map(list, zip(*top_tweet_bigram))

sns.barplot(x, y);
df = pd.concat([train, test], axis=0)

df.shape
def remove_url(text):

    url = re.compile('https?://\S+|www\.\S+')

    return url.sub(r'', text)



# Test the function

remove_url('https://www.kaggle.com/rakkaalhazimi/nlp-disaster-classification/edit?rvi=1')
# Deploy to real data

df['text'] = df['text'].apply(lambda x: remove_url(x))



# Reassure the result

retain = df['text'].str.contains(r'http[s]*').sum()

print("{} words were left behind".format(retain))
# Let's see what the residual words looks like

residual = df[df['text'].str.contains(r'http[s]*')]

left_word = []

for i in range(len(residual)):

    print(residual['text'].values[i])

    left_word.append(residual['text'].values[i])
for word in left_word:

    compiler = re.compile(r'.http.+')

    result = compiler.sub('', word)

    print(result)
# Replace in data frame

df['text'] = df['text'].str.replace(r'.http.+', '')

print("http words found {}".format(df['text'].str.contains('http').sum()))
example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""
def remove_html(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'', text)



print(remove_html(example))
df['text'] = df['text'].apply(lambda x: remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



remove_emoji("Omg another Earthquake 😔😔")
df['text'] = df['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)



example = "I am King#"

remove_punct(example)
df['text'] = df['text'].apply(lambda x: remove_punct(x))
!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spelling(text):

    corrected_text = []

    mispelled_words = spell.unknown(text.split())

    for word in tqdm(text.split()):

        if word in mispelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)



text = "correct me pleasee"

correct_spelling(text)
# df['text'] = df['text'].apply(lambda x: correct_spelling(x))
# df.to_csv("tweetDisaster.csv", index=False)
df = pd.read_csv("/kaggle/input/nlp-disaster-cleaned/tweetDisaster.csv")
# Take a look at 30 samples of data

df['text'].sample(30)
def create_corpus(df):

    corpus = []

    for tweet in tqdm(df['text']):

        words = [word.lower() for word in word_tokenize(tweet) if ((word.isalpha() == 1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus = create_corpus(df)
corpus
embedding_dict = {}

with open("/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt") as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:], 'float32')

        embedding_dict[word] = vectors
print("Embedding shape : ({},{})".format(len(embedding_dict), 

                                         len(embedding_dict['the'])))
MAX_LEN = 50

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences = tokenizer_obj.texts_to_sequences(corpus)



tweet_pad = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
tweet_pad.shape
word_index = tokenizer_obj.word_index

print("Number of unique words:", len(word_index))
word_index
num_words = len(word_index) + 1

embedding_matrix = np.zeros((num_words, 100))



for word, i in tqdm(word_index.items()):

    if i > num_words:

        continue

        

    emb_vec = embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec
embedding_matrix.shape
train = tweet_pad[:train.shape[0]]

test = tweet_pad[train.shape[0]:]
model = Sequential()



embedding = Embedding(num_words, 100, embeddings_initializer=Constant(embedding_matrix),

                      input_length=MAX_LEN, trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.1))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



optimizer = Adam(learning_rate=1e-4)



model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])



model.summary()
history = model.fit(train, target, batch_size=32, epochs=15,

                   validation_split=0.15, verbose=1)
sample_sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_sub.shape
y_pred = model.predict(test)

y_pred = np.round(y_pred).astype(int).reshape(3263)

sub = pd.DataFrame({'id': sample_sub['id'].values.tolist(), 'target': y_pred})

sub.to_csv("submission.csv", index=False)
sub.head()