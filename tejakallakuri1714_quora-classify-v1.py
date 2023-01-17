import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
import nltk
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool2D, Conv2D, GlobalMaxPooling1D, Conv1D, MaxPool1D,MaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)
train_X = pad_sequences(train_X, maxlen=maxlen, padding='pre')
test_X = pad_sequences(test_X, maxlen=maxlen, padding='pre')
train_y = train_df['target'].values
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
#embeddings contains the Word to vector map
#we need to still take care of words that are present in the data but not in the embeddings.
#dict.get() method comes in very handy to avoid "no key found errors"
all_embs = np.stack(embeddings_index.values()) #capture embedding statistics 
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
#defining embedding layer
def pretrained_embedding_layer(embedding_matrix):

    vocab_len = max_features+1 #keras requirement
    embedding_layer = Embedding(vocab_len, embed_size, trainable=False)
    
    # Build the embedding layer, it is required before setting the weights of the embedding layer
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix
    embedding_layer.set_weights([embedding_matrix])
    
    return embedding_layer

def quora_model_v1(input_shape):
    
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(embedding_matrix)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   

    X = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embeddings)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    
    X = Conv1D(32, kernel_size=(7), padding='valid', kernel_initializer='he_uniform')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2)(X)
    
    X = Conv1D(32, kernel_size=(7), padding='valid', kernel_initializer='he_uniform')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2)(X)
    
    X = Bidirectional(CuDNNLSTM(64, return_sequences=False))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    
    X = Dense(128,activation = "relu")(X)
    X = Dropout(0.5)(X)
#     X = Bidirectional(CuDNNLSTM(64, return_sequences=False))(X)
#     X = BatchNormalization()(X)
#     X = Dropout(0.3)(X)
    
    
    X = Dense(1)(X)
    
    out = Activation('sigmoid', name = "final_layer")(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=out)
    
    ### END CODE HERE ###
    
    return model
model = quora_model_v1((maxlen,))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs = 12, batch_size = 512, shuffle=True,validation_split = 0.1)
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_glove_train_y = model.predict([train_X], batch_size=1024, verbose=1)
train_f_score = metrics.f1_score(train_y, (pred_glove_train_y>0.4).astype(int))

pred_test_y = (pred_glove_test_y>0.4).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
