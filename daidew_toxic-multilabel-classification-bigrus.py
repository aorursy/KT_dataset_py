import numpy as np

from tensorflow.keras import layers, models, callbacks, metrics

import tensorflow as tf

from tensorflow.keras.callbacks import Callback,TensorBoard, ModelCheckpoint

from tensorflow.keras.layers import Dense, CuDNNGRU, BatchNormalization, Flatten, Embedding, Bidirectional,TimeDistributed

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import backend as K

from sklearn.metrics import roc_auc_score

import progressbar

import re

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True) #prevent numpy exponential 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename == 'test.csv':

            test = pd.read_csv(os.path.join(dirname, filename))

        elif filename == 'train.csv':

            train = pd.read_csv(os.path.join(dirname, filename))

        elif filename == 'test_labels.csv':

            labels = pd.read_csv(os.path.join(dirname, filename))

        elif filename == 'sample_submission.csv':

            sample = pd.read_csv(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))

        
X,y,X_train,y_train,X_test,y_test = [],[],[],[],[],[]

for i in progressbar.progressbar(range(len(train))):

    l = [train.iloc[i][e] for e in train.keys()]

    X.append(l[1])

    y.append(l[2:])
X_submission = []

for i in progressbar.progressbar(range(len(test))):

    l = [test.iloc[i][e] for e in test.keys()]

    X_submission.append(l[1])
len(X),len(X_submission)
def preprocess_sentence(sen):

    sen = sen.lower().strip()

    sen = re.findall(r"\w+|[^\w\s]", sen, re.UNICODE)

    return sen



def GenerateEmbeddedMatrix(word_model, word_index, sentences,vocab_size):

    embedded_matrix = np.zeros((vocab_size+1, 300))

    print('Word not found in corpus: ')

    cnt = 0

    for word, i in word_index.items():

    #print(word,i)

        if word in word_model.wv:

            embedded_matrix[i] = word_model.wv[word]

        else:

            cnt += 1

        #print(word,end=' ')

    print('\nTotal missing:',cnt,'words')

    return embedded_matrix

def preprocess_data(X):

    for i in range(len(X)):

        try:

            X[i] = preprocess_sentence(X[i])

        except:

            pass

        if len(X[i]) > 100:

            X[i] = X[i][:100]

    for i in range(len(X)):

        for x in range(len(X[i])):

            try:

                X[i][x] = word2num[X[i][x]]

            except:

                X[i][x] = word2num['<UNK>']

    X = pad_sequences(X,maxlen=100,padding='pre',value=0.0)

    return X



class MetricsCallback(Callback):

    def __init__(self, metrics, test_data):

        super().__init__()

        self.X_test,self.y_true = test_data

        self.metrics = metrics

    def on_epoch_end(self, epochs,logs=None):

        roc_aoc_score = self.metrics[0]

        y_pred = self.model.predict(X_test,batch_size=32)

        score = roc_auc_score(self.y_true,y_pred)

        print('\n [roc_auc_score on test set: {}%]'.format(score*100))
mx = max([len(st) for st in X])

mx
for i in range(len(X)):

    X[i] = preprocess_sentence(X[i])

for i in range(len(X_submission)):

    X_submission[i] = preprocess_sentence(X_submission[i])
word2vec_model = Word2Vec(X, size=300, window=5, min_count=1, workers=4)
vocab = set()

word2num = {}

num2word = {}

words = []

k = 0

for i in progressbar.progressbar(range(len(X))):

    for x in range(len(X[i])):

        w = X[i][x]

        if w not in vocab:

            vocab |= {w}

            word2num[w] = k

            num2word[k] = w

            k += 1

words = sorted(list(vocab))

vocab |= {'<UNK>'}

words = ['<UNK>'] + words

word2num['<UNK>'] = k

num2word[k]='<UNK>'

k += 1

words[:10]
embed_matrix = GenerateEmbeddedMatrix(word2vec_model,word2num,X,len(vocab))
_X = preprocess_data(X)

X = _X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
y_train = np.asarray(y_train)

y_test = np.asarray(y_test)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = Sequential()

model.add(Embedding(len(vocab)+1,300,input_length=100,weights=[embed_matrix]))

model.add(Bidirectional(CuDNNGRU(100,return_sequences=True)))

model.add(TimeDistributed(BatchNormalization()))

model.add(Bidirectional(CuDNNGRU(100,return_sequences=True)))

model.add(TimeDistributed(BatchNormalization()))

model.add(Flatten())

model.add(Dense(100,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(6,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.summary()
# Load the extension and start TensorBoard



%load_ext tensorboard.notebook

%tensorboard --logdir=logs/train1

tensorboard_callback = TensorBoard("logs/train1")
filepath = 'models_save/model.h5'

cb = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5)

metrics_callback = MetricsCallback(test_data=(X_test, y_test),

                                   metrics=[roc_auc_score])
model.fit(X_train,y_train,epochs=5,batch_size=64,validation_data=(X_test,y_test),callbacks=[metrics_callback])
model.predict(X_test[20:30],batch_size=10)*100
_X = preprocess_data(X_submission)
X_submission = _X

X_submission[0]
y_submission = model.predict(X_submission, batch_size=64)
res = {}

ids = []

_y = []

toxic,severe_toxic,obscene,threat,insult,identity_hate = [],[],[],[],[],[]

for _id in test['id']:

    ids.append(_id)

toxic = list(y_submission[:,0])

severe_toxic = list(y_submission[:,1])

obscene = list(y_submission[:,2])

threat = list(y_submission[:,3])

insult = list(y_submission[:,4])

identity_hate = list(y_submission[:,5])

res['id'] = ids

res['toxic'] = toxic

res['severe_toxic'] = severe_toxic

res['obscene'] = obscene

res['threat'] = threat

res['insult'] = insult

res['identity_hate'] = identity_hate
df = pd.DataFrame(res, columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
export_csv = df.to_csv ('results.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path