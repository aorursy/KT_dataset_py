import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import scipy as sp

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from keras.models import Model, Input

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

import os

print(os.listdir("../input"))
train_df = pd.read_csv("../input/aij-wikiner-en-wp2", sep=r'\n', header=None, engine='python')

test_df = pd.read_csv("../input/wikigold.conll.txt", sep="\n", header=None, delimiter=" ")

train_df.head()
test_df.columns = ['word', 'iob_tag']
train_df.columns = ['text']

b = pd.DataFrame(train_df.text.str.split(' ').tolist()).stack()

train_df = b.reset_index()

del b

train_df.head()
train_df.columns = ["sentence_id", "no_words_in_sent", "text"]

train_df.head()
train_df[['word','pos_tag', 'iob_tag']] = train_df['text'].str.split('|',expand=True)

train_df.head()
for df in [train_df, test_df]:

    print(df.isna().sum())
test_df.tail()
test_df.drop(test_df.tail(1).index,inplace=True)

test_df.tail()
test_df['sent_end'] = test_df['word'].apply(lambda x: 1 if x[0] =="." else None )
s_end_index = test_df[test_df['sent_end'].notnull()].index.values
test_df['sentence_id'] = None
for i,j in enumerate(s_end_index):

    test_df['sentence_id'].loc[j] = i
test_df['sentence_id'] = test_df['sentence_id'].bfill()
del test_df['sent_end']
train_df['sentence_id'].nunique(), train_df.word.nunique(), train_df.iob_tag.nunique()
train_df.groupby('iob_tag').size().reset_index(name='counts')
train_df.head()
class SentenceGetter(object):

    

    def __init__(self, dataset):

        self.n_sent = 1

        self.dataset = dataset

        self.empty = False

        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),

                                                       #s['pos_tag'].values.tolist(),

                                                        s["iob_tag"].values.tolist())]

        self.grouped = self.dataset.groupby("sentence_id").apply(agg_func)

        self.sentences = [s for s in self.grouped]

    

    def get_next(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None
getter = SentenceGetter(train_df)

sentences = getter.sentences
gt = SentenceGetter(test_df)

test_sentences = gt.sentences
print(sentences[0:1])
maxlen = max([len(s) for s in sentences])

maxlen_test = max([len(s) for s in test_sentences])

print ('Maximum sequence length:', maxlen, maxlen_test)
plt.hist([len(s) for s in sentences], bins=50);

plt.show();
words = list(set(list(train_df["word"].values) + list(test_df['word'].values)))

words.append("ENDPAD")

n_words = len(words); n_words
#tags = list(set(train_df["iob_tag"].values))

tags = list(set(list(train_df["iob_tag"].values) + list(test_df['iob_tag'].values)))

n_tags = len(tags); n_tags
#Converting words to numbers and numbers to words

word2idx = {w: i for i, w in enumerate(words)}

tag2idx = {t: i for i, t in enumerate(tags)}
tag2idx
#padding

X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=maxlen, sequences=X, padding="post",value=n_words - 1)



y = [[tag2idx[w[1]] for w in s] for s in sentences]

y = pad_sequences(maxlen=maxlen, sequences=y, padding="post", value=tag2idx["O"])



X_test = [[word2idx[w[0]] for w in s] for s in test_sentences]

X_test = pad_sequences(maxlen=maxlen, sequences=X_test, padding="post",value=n_words - 1)
y = [to_categorical(i, num_classes=n_tags) for i in y]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X.shape, X_test.shape
input = Input(shape=(maxlen,))

model = Embedding(input_dim=n_words, output_dim=maxlen, input_length=maxlen)(input)

model = Dropout(0.4)(model)

model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

model = Dropout(0.4)(model)

model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)

out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X, np.array(y), batch_size=512, epochs=1, validation_split=0.2, verbose=1)
inv_map = {v: k for k, v in word2idx.items()}

inv_tags= {v: k for k, v in tag2idx.items()}
inv_tags
pred_glove_val_y = model.predict([X_test], batch_size=128, verbose=1)
predictions = []

for i in range(len(pred_glove_val_y)):

    predictions.append(np.argmax(pred_glove_val_y[i], axis=-1))
df = pd.DataFrame()

df['word'] = None

df['tag'] = None

for i in range(len(X_test)):

    dummy = pd.DataFrame()

    ws = list(zip( inv_map[w] for w in X_test[i]))

    preds = list(zip( inv_tags[w] for w in predictions[i]))

    dummy['word'] = np.vstack(ws).tolist()

    dummy['tag'] = np.vstack(preds).tolist()

    dummy['word'] = dummy['word'].str[0]

    dummy['tag'] = dummy['tag'].str[0]

    dummy = dummy[dummy.word != "ENDPAD"]

    df = df.append(dummy)
df['tag'].value_counts()
df.tail()
test_df.tail()
print(confusion_matrix(test_df['iob_tag'], df['tag']))
print(classification_report(test_df['iob_tag'], df['tag']))