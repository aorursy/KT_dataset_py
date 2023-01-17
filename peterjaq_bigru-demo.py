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
import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

from gensim.models import KeyedVectors

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
train = train.fillna(0)
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
#test = test.fillna(0)
sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
#sub = sub.fillna(0)

print('Train shape : ', train.shape)
print('Test shape : ', test.shape)
print('Sub shape:', sub.shape)
train.info()
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  

            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            

            else:
                df[col] = df[col].astype(np.float32)


    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df
train = reduce_mem_usage(train)
fast_text_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
embeddings_index = KeyedVectors.load_word2vec_format(fast_text_common_crawl)
from sklearn import model_selection

#train_df, val = model_selection.train_test_split(train, test_size = 0.1)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import InputSpec, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,  Callback, EarlyStopping, ReduceLROnPlateau
tokenizer = Tokenizer(num_words = 10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(train['comment_text'])

x_train = train['comment_text']
#x_test = val['comment_text']

train_labels = train['target']
#text_labels = val['target']

x_train = tokenizer.texts_to_sequences(x_train)
#x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=256)
#x_test = pad_sequences(x_test, maxlen=256)
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))

num_words_in_embedding = 0

for word, i in tokenizer.word_index.items():
    if word in embeddings_index.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector        
        num_words_in_embedding += 1
def weighted_binary_crossentropy(y_true, y_pred) :

    logloss = -(y_true * K.log(y_pred) * weights[0] + (1 - y_true) * K.log(1 - y_pred) * weights[1])

    return K.mean(logloss, axis=-1)
def build_model(lr=0.0, lr_d=0.0, units=64, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=0, dr=0.1, conv_size=32, epochs=20):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    sequence_input = Input(shape=(256,), dtype='int32')
    embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=256,
                            trainable=False)
    x = embedding_layer(sequence_input)
    x = SpatialDropout1D(spatial_dr)(x)
    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)   
    x = Conv1D(conv_size, kernel_size2, padding = "valid", kernel_initializer = "he_uniform")(x)
    x = Conv1D(int(conv_size/2), kernel_size1, padding = "valid", kernel_initializer = "he_uniform")(x)

    x = Dropout(dr)(x)

    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)     
    
    x = concatenate([avg_pool1, max_pool1])
    x = BatchNormalization()(x)#1
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))#2
    
    preds = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs = sequence_input, outputs = preds)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy","binary_crossentropy"])
    model.summary()
    history = model.fit(x_train, train_labels, batch_size = 512, epochs = epochs, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
   
    model = load_model(file_path)
    return model
Model1= build_model(lr = 1e-3, lr_d = 1e-7, units = 64, spatial_dr = 0.2, kernel_size1=4, kernel_size2=2, dense_units=32, dr=0.1, conv_size=64, epochs=20)
#Model1= build_model(lr = 1e-3, lr_d = 1e-7, units = 64, spatial_dr = 0.2, kernel_size1=4, kernel_size2=2, dense_units=64, dr=0.1, conv_size=64, epochs=20)
#submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

test = test['comment_text']
test = tokenizer.texts_to_sequences(test)
test = pad_sequences(test, maxlen=256)

sub['prediction'] = Model1.predict(test)
sub.reset_index(drop=False, inplace=True)

sub.to_csv('demosubmission.csv', index=False)
