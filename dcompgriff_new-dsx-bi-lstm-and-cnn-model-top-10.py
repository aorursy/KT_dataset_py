import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from nltk.tokenize import TweetTokenizer
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
train.head(10)
train.loc[train.SentenceId == 2]
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))
print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))
print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))
text = ' '.join(train.loc[train.SentenceId == 4, 'Phrase'].values)
text = [i for i in ngrams(text.split(), 3)]
Counter(text).most_common(20)
tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train['Phrase'])
test_vectorized = vectorizer.transform(test['Phrase'])
y = train['Sentiment']
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

import keras
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Input,\
Flatten, Embedding, Conv1D, Bidirectional, MaxPooling1D,\
SpatialDropout1D, Dropout, GaussianDropout, GaussianNoise, \
BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, \
concatenate

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
tk = Tokenizer(lower = True, filters='')
tk.fit_on_texts(full_text)
train_tokenized = tk.texts_to_sequences(train['Phrase'])
test_tokenized = tk.texts_to_sequences(test['Phrase'])
max_len = 50
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embed_size = 300
max_features = 20000
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))
# Simple sequential model.
model = Sequential()
model.add(Embedding(19479, 100, input_length=max_len))
model.add(Bidirectional(GRU(128, return_sequences=False)))
model.add(Dense(64, activation='linear'))

# Compile model
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy', keras.metrics.categorical_accuracy])

# Show the model layout.
model.summary()

# Train the model.
history = model.fit(X_train, y_ohe, batch_size = 1024, epochs = 3, validation_split=0.1, 
                        verbose = 1)

model=Sequential()
model.add(Embedding(19479, 100, input_length=max_len))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Train the model.
history = model.fit(X_train, y_ohe, batch_size = 1024, epochs = 3, validation_split=0.1, 
                        verbose = 1)
# file_path = "best_model.hdf5"
# check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
#                               save_best_only = True, mode = "min")
# early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

# def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
#     inp = Input(shape = (max_len,))
#     x = Embedding(19479, embed_size, weights = [embedding_matrix], trainable = False)(inp)
#     x1 = SpatialDropout1D(dr)(x)

#     x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
#     x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
#     avg_pool1_gru = GlobalAveragePooling1D()(x1)
#     max_pool1_gru = GlobalMaxPooling1D()(x1)
    
#     x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
#     avg_pool3_gru = GlobalAveragePooling1D()(x3)
#     max_pool3_gru = GlobalMaxPooling1D()(x3)
    
#     x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
#     x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
#     avg_pool1_lstm = GlobalAveragePooling1D()(x1)
#     max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
#     x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
#     avg_pool3_lstm = GlobalAveragePooling1D()(x3)
#     max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
#     x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
#                     avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(Dense(128,activation='relu') (x))
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(Dense(100,activation='relu') (x))
#     x = Dense(5, activation = "sigmoid")(x)
#     model = Model(inputs = inp, outputs = x)
#     model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
#     history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 15, validation_split=0.1, 
#                         verbose = 1, callbacks = [check_point, early_stop])
#     model = load_model(file_path)
#     return model
#model = build_model(lr = 1e-4, lr_d = 0, units = 128, dr = 0.5)
pred = model.predict(X_test, batch_size = 1024)
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
# for blending if necessary.
#(ovr.predict(test_vectorized) + svc.predict(test_vectorized) + np.round(np.argmax(pred, axis=1)).astype(int)) / 3
sub['Sentiment'] = predictions
sub.to_csv("blend.csv", index=False)

