# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dframe = pd.read_csv("../input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
# dframe
dframe.columns
dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)
dataset.info()
dataset.head()
dataset=dataset.drop(['shape'],axis=1)
dataset.head()
class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(dataset)
sentences = getter.sentences
print(sentences[5])
maxlen = max([len(s) for s in sentences])
print ('Maximum sequence length:', maxlen)
# Check how long sentences are so that we can pad them
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
plt.hist([len(s) for s in sentences], bins=50)
plt.show()
words = list(set(dataset["word"].values))
words.append("ENDPAD")

n_words = len(words); n_words
tags = list(set(dataset["tag"].values))
n_tags = len(tags); n_tags
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
word2idx['Obama']
tag2idx["O"]
from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=140, sequences=X, padding="post",value=n_words - 1)
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])
from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
input = Input(shape=(140,))
model = Embedding(input_dim=n_words, output_dim=140, input_length=140)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=1, validation_split=0.2, verbose=1)
i = 0
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
print("{:14} {}".format("Word", "Pred"))
for w,pred in zip(X_test[i],p[0]):
    print("{:14}: {}".format(words[w],tags[pred]))

p[0]
model.summary()


tweet = pd.read_csv("../input/entweets/EnglishTweets.csv")
tweet = tweet.drop(['Unnamed: 1'], axis=1)
# tweet.text
import re
def remove_hypelinks(tweet):
    return re.sub(r'https?:.*[\r\n]*', '', tweet)

def remove_hastags(tweet):
    tweet2 = re.sub(r'^RT[\s]+', '', tweet)
    tweet2 = re.sub(r'#', '', tweet2)
    return tweet2

def remove_mention(tweet):
    tweet  = re.sub(r'@\w+ ?', '', tweet)
    return tweet

EMOJIS = [[':)', '😀'],[';)', '😉'],[':(', '😞'],[';((', '😢'],[':P', '😛'],[':D', '😀']]
_emoji_re = '[\U00010000-\U0010ffff]+'
emoji_re = re.compile(_emoji_re, flags=re.UNICODE)


def emoji_normalize(text):
    for e1, e2 in EMOJIS:
        text = text.replace(e1, e2)
    return text


def remove_emoji(text):
    text = emoji_normalize(text)
    text =   emoji_re.sub('', text)
    return re.sub('\s+', ' ', text)

def remove_extras(text):
    text = re.sub(r'\.\.\.', '', text)
    text = re.sub(r'\.\.','', text)
    text = re.sub(r'\:', '', text)
    return text 


def preprocess(tweet):
        tweet = remove_hypelinks(tweet) or tweet
        tweet = remove_mention(tweet) or tweet 
        tweet = remove_hastags(tweet) or tweet
        tweet = remove_emoji(tweet) or tweet
        tweet = remove_extras(tweet) or tweet
        return tweet

# tweet = "American Airlines said it would launch a direct flight to Bengaluru from Seattle :D, home to Amazon and Microsoft https:xyz.com."
# print(tweet)
print(tweet.text[2])
preprocess(tweet.text[2])

tweets = [preprocess(sent) for sent in tweet.text]
tweets[2]
tweet['clean'] = tweets

tweet.sample(10)
lst = [s.split() for s in tweets]


# 'no' in word2idx
# '-' as 'UNK'
x_1 = [[word2idx[w] if w in word2idx else word2idx['-'] for w in s] for s in lst]
x_1 = pad_sequences(maxlen=140, sequences=x_1, padding="post",value=n_words - 1)
i = 1
p = model.predict(np.array([x_1[i]]))
p = np.argmax(p, axis=-1)
print("{:14} {}".format("Word", "Pred"))
for w,pred in zip(x_1[i],p[0]):
    if tags[pred] != 'O':
        print((words[w],tags[pred]))
def predict(i):
    p = model.predict(np.array([x_1[i]]))
    p = np.argmax(p, axis=-1)
    out = []
    for w, pred in zip(x_1[i], p[0]):
        if tags[pred] != 'O':
            out.append((words[w], tags[pred]))
    return out
    
len(tweet.clean)
[predict(i) for i in range(len(tweet.clean))]
tweet['extracted_entities'] = [predict(i) for i in range(len(tweet.clean))]
tweet.head(10)
