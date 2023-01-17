import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, BatchNormalization
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
train.head(10)
test.head(10)
sub.head(10)
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

tokenizer = Tokenizer(lower = True, filters = '')
tokenizer.fit_on_texts(full_text)
tokenizer.word_index
train_tokenized = tokenizer.texts_to_sequences(train['Phrase'])
test_tokenized = tokenizer.texts_to_sequences(test['Phrase'])
max_len = 50
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)
print(X_train)
print(X_test)
embed_size = 300
max_features = 20000
def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')
def get_embed_mat(embedding_path):
    
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix
y = train['Sentiment']

one_hot_encoder = OneHotEncoder(sparse=False)
y_one_hot = one_hot_encoder.fit_transform(y.values.reshape(-1, 1))
file_path = "model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)
def rnn_model(emb_input, embedding_matrix):
    
    input_layer = Input(shape = (max_len,))
    x = Embedding(emb_input, embed_size, weights = [embedding_matrix], trainable = False)(input_layer)
    x1 = SpatialDropout1D(0.2)(x)
    
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x = Bidirectional(GRU(128, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
    y = Bidirectional(LSTM(256, return_sequences = True))(x1)
    y = Bidirectional(LSTM(128, return_sequences = True))(y)
    y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)
    
    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)
    
    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)
    
    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
    x = BatchNormalization()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(5, activation = "sigmoid")(x)
    model = Model(inputs = input_layer, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 0.001, decay = 0), metrics = ["accuracy"])
    history = model.fit(X_train, y_one_hot, batch_size = 128, epochs = 100, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model    
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embedding_matrix = get_embed_mat(embedding_path)
emb_input = embedding_matrix.shape[0]
model = rnn_model(emb_input, embedding_matrix)
pred = model.predict(X_test, batch_size = 1024, verbose = 1)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
sub['Sentiment'] = np.round(predictions).astype(int)
sub.to_csv("output.csv", index=False)

