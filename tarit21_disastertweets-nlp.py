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
import pandas as pd

import numpy as np

import sys

sys.path.append('../')

import seaborn as sns

import matplotlib.pyplot as plt

import re

import string

import eli5



from string import punctuation

from collections import defaultdict

from nltk import FreqDist

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import CountVectorizer

#from src.preprocessing.text import *
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test  = pd.read_csv('../input/nlp-getting-started/test.csv')

gt = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
x=train.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

tweet_len = train[train['target']==1]['text'].str.len()

ax1.hist(tweet_len, color='green')

ax1.set_title('Disaster tweets')



tweet_len = train[train['target']==0]['text'].str.len()

ax2.hist(tweet_len, color='red')

ax2.set_title('Not disaster tweets')



fig.suptitle('Characters in tweets')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

num_words = train[train['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(num_words, color='red')

ax1.set_title('Disaster tweets')



num_words = train[train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(num_words, color='green')

ax2.set_title('Non disaster tweets')



fig.suptitle('Words in a tweets')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

word = train[train['target']==1]['text'].str.split().apply(lambda x: [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='red')

ax1.set_title('Disaster tweets')



word = train[train['target']==0]['text'].str.split().apply(lambda x: [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='green')

ax2.set_title('Non disaster tweets')



fig.suptitle('Average word length in each tweet')
text_disaster = train[train['target']==1]['text'].str.split()

text_Nodisaster = train[train['target']==0]['text'].str.split()
fdist = FreqDist(word.lower() for sentence in text_disaster for word in sentence)

fdist.plot(10, title="Disaster tweets")



dic=defaultdict(int)

punct = [fdist[p] for p in punctuation]

plt.figure(figsize=(12, 6))

sns.barplot(punct, list(punctuation))
fdist = FreqDist(word.lower() for sentence in text_Nodisaster for word in sentence)

fdist.plot(10, title="Non disaster tweets")



dic=defaultdict(int)

punct = [fdist[p] for p in punctuation]

plt.figure(figsize=(12, 6))

sns.barplot(punct, list(punctuation))
def get_top_tweet_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_tweet_bigram = get_top_tweet_bigram(train['text'].tolist())[:10]

x,y = map(list, zip(*top_tweet_bigram))

sns.barplot(y,x)
train = train.fillna('.')

test = test.fillna('.')
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
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
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
def preprocessing(re_URL=False, re_emoji=False, re_punct=False):

    data_train = train['keyword'] +' '+ train['location'] +' '+ train['text']

    data_test = test['keyword'] +' '+ test['location'] +' '+ test['text']



    if re_URL:

        data_train = data_train.apply(lambda x : remove_URL(x))

        data_test = data_test.apply(lambda x : remove_URL(x))

        print("URL Removed")

    if re_emoji:

        data_train = data_train.apply(lambda x : remove_emoji(x))

        data_test = data_test.apply(lambda x : remove_emoji(x))

        print("Emoji Removed")

    if re_punct:

        data_train = data_train.apply(lambda x : remove_punct(x))     

        data_test = data_test.apply(lambda x : remove_punct(x))

        print("Punctuation Removed")

    return data_train, data_test
def fit_and_predict(vec, clf, X_train, y_train):

    pipe = make_pipeline(vec, clf)

    pipe.fit(X_train, y_train)

    

    y_test = gt['target'].tolist()

    acc = pipe.score(X_test, y_test)

    print("Accuracy: ", acc)
train['sums'] = train['keyword'] +' '+ train['location'] +' '+ train['text']

test['sums'] = test['keyword'] +' '+ test['location'] +' '+ test['text']



X_train = train['sums'].tolist()

y_train = train['target'].tolist()

X_test = test['sums'].tolist()



vec = CountVectorizer(ngram_range=(1,2))

clf = LogisticRegression()

fit_and_predict(vec, clf, X_train, y_train)

train['sums'] = train['keyword'] +' '+ train['location'] +' '+ train['text']

test['sums'] = test['keyword'] +' '+ test['location'] +' '+ test['text']



X_train = train['sums'].tolist()

y_train = train['target'].tolist()

X_test = test['sums'].tolist()



vec = CountVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')

clf = LogisticRegression()

fit_and_predict(vec, clf, X_train, y_train)
train['sums'], test['sums'] = preprocessing(re_URL=True, re_emoji=True, re_punct=True)

X_train = train['sums'].tolist()

y_train = train['target'].tolist()

X_test = test['sums'].tolist()



vec = CountVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')

clf = LogisticRegression()

fit_and_predict(vec, clf, X_train, y_train)
train['sums'], test['sums'] = preprocessing(re_URL=True, re_emoji=False, re_punct=True)

X_train = train['sums'].tolist()

y_train = train['target'].tolist()

X_test = test['sums'].tolist()



vec = CountVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')

clf = LogisticRegression()

fit_and_predict(vec, clf, X_train, y_train)
train['sums'], test['sums'] = preprocessing(re_URL=False, re_emoji=False, re_punct=True)

X_train = train['sums'].tolist()

y_train = train['target'].tolist()

X_test = test['sums'].tolist()



vec = CountVectorizer(ngram_range=(1,2), lowercase=True, stop_words='english')

clf = LogisticRegression()

fit_and_predict(vec, clf, X_train, y_train)
eli5.show_weights(clf, vec=vec, top=10)
eli5.show_prediction(clf, X_test[0], vec=vec, target_names=['0', '1'])
eli5.show_prediction(clf, X_test[-2], vec=vec, target_names=['0', '1'])
eli5.show_weights(clf, vec=vec, top=10)
eli5.show_prediction(clf, X_test[0], vec=vec, target_names=['0', '1'])
eli5.show_prediction(clf, X_test[-2], vec=vec, target_names=['0', '1'])
eli5.show_weights(clf, vec=vec, top=10)
eli5.show_prediction(clf, X_test[0], vec=vec, target_names=['0', '1'])
eli5.show_prediction(clf, X_test[-2], vec=vec, target_names=['0', '1'])
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

import string





def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'', text)





def remove_html(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'', text)





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





def remove_punctuation(text):

    table = str.maketrans('', '', string.punctuation)

    return text.translate(table)
def clean_wrapper(text): 

    text = remove_url(text)

    text = remove_html(text)

    text = remove_emoji(text)

    text = remove_punctuation(text)

    return text
train['text'] = train['text'].apply(lambda x : clean_wrapper(x))

test['text'] = test['text'].apply(lambda x : clean_wrapper(x))
sent_data = train.text.values

labels_data = train.target.values

sent_submission = test.text.values
from sklearn.model_selection import train_test_split

sent_train, sent_test, labels_train, labels_test = train_test_split(sent_data, labels_data, test_size=0.2, random_state=42)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(sent_data)



X_train = tokenizer.texts_to_sequences(sent_data)

# X_test = tokenizer.texts_to_sequences(sent_test)

X_submission = tokenizer.texts_to_sequences(sent_submission)



y_train = labels_data

# y_test = labels_test



vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index



print(sent_train[2])

print(X_train[2])
maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

X_submission = pad_sequences(X_submission, padding='post', maxlen=maxlen)



print(X_train[0, :])
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
from tensorflow.keras import Sequential, layers, regularizers

embedding_dim = 100

drop_out_prob = 0.3



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(30, activation='relu'))

model.add(layers.Dropout(drop_out_prob))

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dropout(drop_out_prob))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=1,

                    validation_split=0.2,

                    batch_size=100)

plot_history(history)