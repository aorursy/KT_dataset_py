import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.layers import concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv')
dftest=pd.read_csv('../input/test.csv')
print(df.shape, dftest.shape)

train_df, val_df = train_test_split(df, test_size=0.1, random_state=1995)

train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = dftest["question_text"].fillna("_na_").values

embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use
#df.astype(int64)
#df.describe().apply(lambda x: format(x,  '%.3f' % x))
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
train_y = train_df['target'].values
val_y = val_df['target'].values
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_X.shape
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False
    
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
import gc
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_1[i] = embedding_vector

del embeddings_index; gc.collect() 
def model2():
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size, weights=[embedding_matrix_1], trainable=False)(inp)
    x = embed
    
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = AttLayer(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    

    return model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import Add, BatchNormalization, Activation, CuDNNLSTM, Dropout
from keras.layers import *
from keras.models import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
from sklearn import metrics









F1scores_epochs = [0.6327, 0.6607, 0.6688]


F1scores_epochs = [0.6327, 0.6607, 0.6688, 0.6698]
MODEL2 = model2()
MODEL2.summary()

batch_size = 2048
epochs = 5

early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
model_checkpoint = ModelCheckpoint('./model2.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

hist = MODEL2.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_y), verbose=True)
MODEL2.save('./model2.h5')
pred_val_y_2 = MODEL2.predict([val_X], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y_2 > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_2 = thresholds[0][0]
print("Best threshold: ", best_thresh_2)

y_pred_2 = MODEL2.predict(test_X, batch_size=1024, verbose=True)
print(y_pred_2)
import matplotlib.pyplot as plt
%matplotlib inline
Order = [1, 2, 3, 4]
F1scores_epochs = [0.6327, 0.6607, 0.6688, 0.6698]

LABELS = ["1", "2","3", "5"]
plt.figure(figsize = (10,5))
plt.bar(Order, F1scores_epochs, align='center', width=0.3)
plt.xticks(Order, LABELS)
plt.xlabel('Epochs')
plt.ylabel('F-1 scores')
plt.ylim(0.63, 0.675)
plt.show()
Order = [1, 2, 3, 4]
Validation_Accuracy = [0.9546, 0.9572, 0.957, 0.6698]

LABELS = ["1", "2","3", "5"]
plt.figure(figsize = (10,5))
plt.bar(Order, F1scores_epochs, align='center', width=0.3)
plt.xticks(Order, LABELS)
plt.xlabel('Epochs')
plt.ylabel('F-1 scores')
plt.ylim(0.63, 0.675)
plt.show()
y_te = (y_pred_2[:,0] > best_thresh_2).astype(np.int)

submit_df = pd.DataFrame({"qid": dftest["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)
#sub = pd.read_csv('../input/sample_submission.csv')
#sub.to_csv("submission.csv", index=False)

