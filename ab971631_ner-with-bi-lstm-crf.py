import pandas as pd

import numpy as np

import re

import string



data = pd.read_csv("../input/entity-annotated-corpus/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
df1 = pd.DataFrame({ "Sentence #":['Sentence: 47960']*6, 

                    "Word":['my', 'name', 'is', 'abhishek', 'kumar','.'],  

                    "POS":[None]*6,

                    "Tag":['O','O','O','B-per','I-per','O']})

df2 = pd.DataFrame({ "Sentence #":['Sentence: 47961']*7, 

                    "Word":['my', 'name', 'is', 'ritik', 'kumar','gupta','.'],  

                    "POS":[None]*7,

                    "Tag":['O','O','O','B-per','I-per','I-per','O']})
data=data.append(df1)

data=data.append(df2)
data.tail()
data.head(20)
words = list(set(data["Word"].values))

words.append("ENDPAD")

n_words = len(words); n_words
tags = list(set(data["Tag"].values))

n_tags = len(tags); n_tags
class SentenceGetter(object):

    

    def __init__(self, data):

        self.n_sent = 1

        self.data = data

        self.empty = False

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),

                                                           s["POS"].values.tolist(),

                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(agg_func)

        self.sentences = [s for s in self.grouped]

    

    def get_next(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)
sentences = getter.sentences
max_len = 75

word2idx = {w: i + 1 for i, w in enumerate(words)}

tag2idx = {t: i for i, t in enumerate(tags)}
word2idx["Obama"]
tag2idx["B-geo"]
from keras.preprocessing.sequence import pad_sequences



# pad the sequence

X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)
# pad the target

y = [[tag2idx[w[2]] for w in s] for s in sentences]

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
from keras.utils import to_categorical

y = [to_categorical(i, num_classes=n_tags) for i in y]

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
!pip install git+https://www.github.com/keras-team/keras-contrib.git
from keras.models import Model, Input,Sequential

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from keras_contrib.layers import CRF

import keras as k
model = Sequential()

model.add(Embedding(input_dim=n_words+1, output_dim=200, input_length=max_len))

model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1)))

model.add(TimeDistributed(Dense(n_tags, activation="relu")))

crf_layer = CRF(n_tags)

model.add(crf_layer)
model.summary()
# adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

model.compile(optimizer='adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
history = model.fit(X_tr, np.array(y_tr), batch_size=128, epochs=5,

                    validation_split=0.1, verbose=1)
hist = pd.DataFrame(history.history)
import matplotlib.pyplot as plt

plt.style.use("ggplot")

plt.figure(figsize=(12,12))

plt.plot(hist["crf_viterbi_accuracy"])

plt.plot(hist["val_crf_viterbi_accuracy"])

plt.show()
!pip install seqeval
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
test_pred = model.predict(X_te, verbose=1)
idx2tag = {i: w for w, i in tag2idx.items()}



def pred2label(pred):

    out = []

    for pred_i in pred:

        out_i = []

        for p in pred_i:

            p_i = np.argmax(p)

            out_i.append(idx2tag[p_i].replace("PAD", "O"))

        out.append(out_i)

    return out

    

pred_labels = pred2label(test_pred)

test_labels = pred2label(y_te)
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

print(classification_report(test_labels, pred_labels))
model.evaluate(X_te, np.array(y_te))
i = 1927

p = model.predict(np.array([X_te[i]]))

p = np.argmax(p, axis=-1)

true = np.argmax(y_te[i], -1)

print("{:15}||{:5}||{}".format("Word", "True", "Pred"))

print(30 * "=")

for w, t, pred in zip(X_te[i], true, p[0]):

    if w != 0:

        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))
# Custom Tokenizer

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
test_sentence="In the context of prehistory, antiquity and contemporary indigenous peoples, the title may refer to tribal kingship. Germanic kingship is cognate with Indo-European traditions of tribal rulership (c.f. Indic rājan, Gothic reiks, and Old Irish rí, etc.)."

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]],

                            padding="post", value=0, maxlen=max_len)
p = model.predict(np.array([x_test_sent[0]]))

p = np.argmax(p, axis=-1)

print("{:15}||{}".format("Word", "Prediction"))

print(30 * "=")

for w, pred in zip(tokenize(test_sentence), p[0]):

    print("{:15}: {:5}".format(w, tags[pred]))