# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , LSTM , Embedding , Conv1D , Bidirectional , GRU , Dropout
from keras.layers import GlobalMaxPool1D, Dropout, Activation,CuDNNLSTM
from keras.layers import MaxPooling1D, BatchNormalization,Conv2D,Flatten
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
# Training and Test set processing
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_data, validation_data = train_test_split(train, test_size=.1, random_state=1234)
embed_size = 300 # how big is each word vector
max_features = 900000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

train_X = train_data["question_text"].fillna("_na_").values
val_X = validation_data["question_text"].fillna("_na_").values
test_X = test["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X)+list(val_X)+list(test_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X,maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_data['target'].values
val_y = validation_data['target'].values
print("Done")
import gc
#Using Embeddings
embedding_index = dict()
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt',encoding='utf8')

for line in f:
    values = line.split(" ")
    words = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[words]= coefs
    
f.close()
embedding_matrix_glove = np.zeros((max_features, 300))
for word, index in tokenizer.word_index.items():
    if index > max_features - 1:
        break
    else:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_glove[index] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix_glove, axis=1) == 0))
del embedding_index; gc.collect()  
from keras import backend as K
model=Sequential()
model.add(Embedding(max_features, 300, weights=[embedding_matrix_glove],input_length=maxlen,trainable=False) )
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
model.add(Attention(maxlen))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Done")
#training
model.fit(train_X,train_y,epochs=2,batch_size=512)
print("Done")
from sklearn import metrics
pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

best_thresh = 0.5
best_score = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    if score > best_score:
        best_thresh = thresh
        best_score = score

print("Val F1 Score: {:.4f}".format(best_score))
best_thresh
pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_test_y>0.41)                                    #changing the threshold in this version
out_df = pd.DataFrame({"qid":test["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
