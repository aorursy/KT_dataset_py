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
train_df = pd.read_csv('../input/train.csv')
X_train = train_df["question_text"].fillna("dieter").values
test_df = pd.read_csv('../input/test.csv')
X_test = test_df["question_text"].fillna("dieter").values
y = train_df["target"]
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
maxlen = 70
max_features = 50000
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
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
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_2[i] = embedding_vector
del embeddings_index; gc.collect()
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_3[i] = embedding_vector
        
del embeddings_index; gc.collect()   
# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
from gensim.models import KeyedVectors

EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix_4 = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_features: continue
    if word in embeddings_index:
        embedding_vector = embeddings_index.get_vector(word)
        embedding_matrix_4[i] = embedding_vector
        
del embeddings_index; gc.collect()        
embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4), axis=1)  
del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4
gc.collect()
np.shape(embedding_matrix)

def model1(init):
    x = Reshape((maxlen, embed_size * 4, 1))(init)
    
    filter_sizes = [1,2,3]
    num_filters = 32
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), padding='valid', kernel_initializer='normal', activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), padding='valid', kernel_initializer='normal', activation='relu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), padding='valid', kernel_initializer='normal', activation='relu')(x)
    #conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), padding='valid', kernel_initializer='normal', activation='relu')(x)

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
    #maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1), strides=(1,1), padding='valid')(conv_3)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    
    gmp = GlobalMaxPooling2D()(concatenated_tensor)
    gap = GlobalAveragePooling2D()(concatenated_tensor)
    
    conc = Concatenate(axis=1)([gmp, gap])
    
    #flatten = Flatten()(concatenated_tensor)
    return conc

def model2(init):
    x = init
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = AttLayer(64)(x)
    return x


def model3(init):
    filters = 64
    x = init
    pool = []
    
    x = Conv1D(filters, 1, activation='relu')(init)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.1)(x)
    
    x = Conv1D(filters, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.1)(x)
    
    x = Conv1D(filters, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.1)(x)
    
    x = Conv1D(filters, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.1)(x)
    
    x = Flatten()(x)
    return x

def model4(init):
    x = init
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = AttLayer(64)(x)
    return x

def all_embed(init):
    x = init
    
    #out1 = model1(x)
    #out2 = model2(x)    #CuDNNLSTM
    out3 = model3(x)   #Conv1D
    #out4 = model4(x)    #CuDNNGRU
    
    #conc = concatenate([out1, out2, out3, out4])
    
    return out3
    
def get_model():
    inp = Input(shape=(maxlen, ))
    embed1 = Embedding(max_features, embed_size * 4, weights=[embedding_matrix], trainable=False)(inp)
    
    out = all_embed(embed1)
    
    x = Dropout(0.3)(out)
    x = Dense(128, activation='relu')(x)
    
    outp = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    

    return model
model = get_model()
model.summary()
from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)
early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
model_checkpoint = ModelCheckpoint('./quora.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

#model = load_model('./quora.model')

batch_size = 1536
epochs = 3

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
pred_val_y = model.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)
y_pred = model.predict(x_test, batch_size=1024, verbose=True)
y_te = (y_pred[:,0] > best_thresh).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)
from IPython.display import HTML
import base64  
import pandas as pd  

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submit_df)
