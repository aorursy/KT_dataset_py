!pip install -q git+https://www.github.com/keras-team/keras-contrib.git

!pip install -q seqeval
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
train_path = '../input/data/train.csv'

test_path = '../input/data/test.csv'
df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)

print(df.shape, test_df.shape)
df.apply(lambda x: len(x.unique()))


df['tag'].value_counts()
df.head()
df = train_df.fillna(method="ffill")

df.isnull().sum()
test_df = test_df.fillna(method="ffill")

test_df.isnull().sum()
def sentence_getter(data):

    sentences = data.groupby("Sent_ID").apply(lambda d: [w for w in d["Word"].values.tolist()]).tolist()

    return sentences



def tags_getter(data):

    tags = data.groupby("Sent_ID").apply(lambda d: [t for t in d["tag"].values.tolist()]).tolist()

    return tags

    
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



class preprocess_data(object):

    

    def __init__(self, max_len):

        self.max_len = max_len

        

        self.vocab = None

        self.tags = None

        

        self.word2idx = None

        self.idx2word = None

        

        self.tag2idx = None

        self.idx2tag = None



        

    def fit_transform(self, train_sent, train_tags):

        self.vocab = list(set([word for sublist in train_sent for word in sublist]))

        self.tags = list(set([tag for sublist in train_tags for tag in sublist]))

        

        self.word2idx = {w: i + 2 for i, w in enumerate(self.vocab)}

        self.word2idx["UNK"] = 1

        self.word2idx["PAD"] = 0

        self.idx2word = {i: w for w, i in self.word2idx.items()}

        

        self.tag2idx = {t: i for i, t in enumerate(self.tags)}

        # self.tag2idx["PAD"] = 0

        self.idx2tag = {i: w for w, i in self.tag2idx.items()}

        

        X = [[self.word2idx[w] for w in s] for s in train_sent]

        y = [[self.tag2idx[i] for i in t] for t in train_tags]

        

        X = pad_sequences(maxlen= self.max_len, sequences= X, padding="post", value= self.word2idx["PAD"])

        y = pad_sequences(maxlen= self.max_len, sequences=y, padding="post", value= self.tag2idx["O"])

        y = np.array([to_categorical(i, num_classes=len(self.tag2idx)) for i in y])

        

        return X,y

        

        

    def transform(self, test_sent, test_tags = None):

        X_test = []

        for sentence in test_sent:

            

            sentence_idx = []

            for word in sentence:

                try:

                    widx = self.word2idx[word] 

                except:

                    widx = self.word2idx["UNK"]

                sentence_idx.append(widx)

                

            X_test.append(sentence_idx)

            

        X_test = pad_sequences(maxlen= self.max_len, sequences= X_test, padding="post", value= self.word2idx["PAD"])

            

        if test_tags != None:

            y_test = [[self.tag2idx[i] for i in t] for t in test_tags]

            y_test = pad_sequences(maxlen= self.max_len, sequences=y_test, padding="post", value= self.tag2idx["O"])

            y_test = np.array([to_categorical(i, num_classes=len(self.tag2idx)) for i in y_test])

            return X_test, y_test

        

        return X_test
train_df = df[:int(len(df) * 0.8)]

val_df = df[int(len(df) * 0.8):]
train_sent, train_tags = sentence_getter(train_df), tags_getter(train_df)

val_sent, val_tags = sentence_getter(val_df), tags_getter(val_df)

test_sent =  sentence_getter(test_df)
len(train_sent), len(train_tags), len(val_sent), len(val_tags), len(test_sent)
lens = [len(s) for s in train_sent]

print(max(lens), min(lens), sum(lens)/len(lens), np.median(lens))



import matplotlib.pyplot as plt

plt.style.use("ggplot")

plt.hist(lens, bins=200)

plt.show()
max_len = 25

ner = preprocess_data(max_len = 25)

x_train, y_train = ner.fit_transform(train_sent, train_tags)

x_val, y_val = ner.transform(val_sent, val_tags)

x_test =  ner.transform(test_sent)
x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape
from keras.models import Model, Input

from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, CuDNNLSTM

from keras_contrib.layers import CRF

from keras import losses, metrics





inp = Input(shape=(max_len,))

model = Embedding(input_dim=len(ner.vocab), output_dim=64, input_length=max_len)(inp)

model = Dropout(0.1)(model)

model = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(model)

model = TimeDistributed(Dense(16, activation="relu"))(model)

crf = CRF(len(ner.tag2idx))  # CRF layer

out = crf(model)  # output
model = Model(inp, out)

model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data = (x_val, y_val))
idx2tag = ner.idx2tag

def pred2label(pred):

    out = []

    for pred_i in pred:

        out_i = []

        for p in pred_i:

            p_i = np.argmax(p)

            out_i.append(idx2tag[p_i].replace("PAD", "O"))

        out.append(out_i)

    return out
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

val_pred = model.predict(x_val, verbose = 1)

val_labels_true = pred2label(y_val)

val_labels_pred = pred2label(val_pred)

print()

print("F1-score: {:.1%}".format(f1_score(val_labels_true, val_labels_pred)))

print()

print(classification_report(val_labels_true, val_labels_pred))
test_df.head()
test_pred = model.predict(x_test, verbose = 1)

test_labels_pred = pred2label(test_pred)
sent_ids = test_df['Sent_ID'].unique()
res = []

for _sent, _labels, _sentid in zip(x_test,test_labels_pred, sent_ids):

    for _idx, _label in zip(_sent, _labels):

        res.append((_sentid, ner.idx2word[_idx], _label))
res_df = pd.DataFrame(data = res, columns = ['Sent_ID', 'Word', 'tag'])

res_df = res_df[~(res_df['Word'] == 'PAD')]
res_df['tag'].value_counts()
sub_df = pd.merge(test_df[['id','Sent_ID', 'Word']].fillna('ffill'), res_df, how='left', on = ['Sent_ID', 'Word'])

sub_df['tag'] = sub_df['tag'].fillna('O')

sub_df = sub_df[~(sub_df.duplicated(keep = 'first'))]

sub_df = sub_df[~(sub_df['Word'] == 'UNK')]

sub_df = sub_df[~(sub_df['id'].duplicated())]

print(test_df.shape, sub_df.shape)
sub_df['tag'].value_counts()
sub_df.to_csv('sub01.csv', index = None)

from IPython.display import FileLink

FileLink(f'sub01.csv')