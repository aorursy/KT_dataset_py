import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, SimpleRNN, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
from tqdm.notebook import tqdm
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
train1_csv = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train1 = train1_csv[['comment_text', 'toxic']]
display(train1.head())
print(train1.shape)
train1.toxic.value_counts(normalize=True)
# train2_csv = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
# train2_high = train2_csv[['comment_text', 'toxic']][train2_csv.toxic > 0.80].reset_index(drop=True)
# train2_low = train2_csv[['comment_text', 'toxic']][train2_csv.toxic == 0.00]\
#             .sample(frac=0.05, replace=False, random_state=42).reset_index(drop=True)
# display(train2_high.head())
# print("toxic > 0.75", train2_high.shape)
# display(train2_low.head())
# print("toxic = 0.00", train2_low.shape)
# train = pd.concat([train1, train2_high, train2_low], ignore_index=True)
# display(train.head())
# print(train.shape)
valid = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")
valid = valid[['comment_text', 'lang', 'toxic']]
display(valid.head())
valid.lang.value_counts(normalize=True)
# Combine training data with non-english samples from valid
valid_samples = valid.groupby('lang', group_keys=False)\
                     .apply(lambda g: g.sample(frac=0.5, random_state=1))\
                     .reset_index(drop=True)
valid_samples
train = pd.concat([train1, valid_samples], ignore_index=True)
train
cond = valid.comment_text.isin(valid_samples.comment_text)
valid.drop(valid[cond].index, inplace = True)
valid
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train.comment_text.values)
sequences = tokenizer.texts_to_sequences(train.comment_text.values)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)
train.comment_text.iloc[213924]
sequence_to_text(sequences[213924])
valid_sequences = tokenizer.texts_to_sequences(valid.comment_text.values)
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
train_padded[0]
sequence_to_text(train_padded[0])
# output directory
OUTPUT_DIR = '../input/output/dense'

# training
EPOCHS = 4
BATCH_SIZE = 128

# embedding
N_DIM = 64
N_UNIQUE_WORDS = 5000
N_WORDS_TO_SKIP = 50
MAX_LENGTH = 100
PAD_TYPE = TRUNC_TYPE = 'pre'

# NN architecture
N_DENSE = 64
DROPOUT = 0.5
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(Flatten())
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
gc.collect()
modelcheckpoint = ModelCheckpoint("/kaggle/working/dense/best.hdf5", \
                                  monitor='valloss', savebest_only=True)
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
y_hat = model.predict_proba(valid_padded)
plt.hist(y_hat)
_ = plt.axvline(x=0.5, color='orange')
from tensorflow.keras.layers import SpatialDropout1D, Conv1D, GlobalMaxPooling1D
# training:
EPOCHS = 4
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 20000 
MAX_LEN = 400
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_CONV = 256 
K_CONV = 3 

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(Conv1D(N_CONV, K_CONV, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
# training:
EPOCHS = 16
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 20000 
MAX_LEN = 100
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_RNN = 256 
DROP_RNN = 0.2

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(SimpleRNN(N_RNN, activation='relu'))
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
from tensorflow.keras.layers import GRU
# training:
EPOCHS = 4
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 20000 
MAX_LEN = 100
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_GRU = 256 
DROP_GRU = 0.2

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(GRU(N_GRU, dropout=DROP_GRU))
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
from tensorflow.keras.layers import LSTM
# training:
EPOCHS = 4
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 20000 
MAX_LEN = 100
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_LSTM = 256 
DROP_LSTM = 0.2

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(LSTM(N_LSTM, dropout=DROP_LSTM))
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
from tensorflow.keras.layers import Bidirectional
# training:
EPOCHS = 6
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 20000 
MAX_LEN = 200
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_LSTM = 256 
DROP_LSTM = 0.2

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(Bidirectional(LSTM(N_LSTM, dropout=DROP_LSTM)))
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
# training:
EPOCHS = 4
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 20000 
MAX_LEN = 200
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_LSTM = 256
DROP_LSTM = 0.2

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(Bidirectional(LSTM(N_LSTM, dropout=DROP_LSTM, return_sequences=True)))
    model.add(Bidirectional(LSTM(N_LSTM, dropout=DROP_LSTM)))
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
from tensorflow.keras.layers import MaxPooling1D
# training:
EPOCHS = 4
BATCH_SIZE = 128

# vector-space embedding: 
N_DIM = 64
N_UNIQUE_WORDS = 40000 
MAX_LEN = 200
PAD_TYPE = TRUNC_TYPE = 'pre'
DROP_EMBED = 0.2 

# convolutional layer architecture:
N_CONV = 64
K_CONV = 3
MP_SIZE = 4

# LSTM layer architecture:
N_LSTM = 64
DROP_LSTM = 0.2

# dense layer architecture: 
N_DENSE = 256
DROPOUT = 0.2
train_padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
valid_padded = pad_sequences(valid_sequences, maxlen=MAX_LENGTH, padding=PAD_TYPE, \
                             truncating=TRUNC_TYPE, value=0)
with strategy.scope():
    model = Sequential()
    model.add(Embedding(N_UNIQUE_WORDS, N_DIM, input_length=MAX_LENGTH))
    model.add(SpatialDropout1D(DROP_EMBED))
    model.add(Conv1D(N_CONV, K_CONV, activation='relu'))
    model.add(MaxPooling1D(MP_SIZE))
    model.add(Bidirectional(LSTM(N_LSTM, dropout=DROP_LSTM)))
    model.add(Dense(N_DENSE, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_padded, train.toxic.values, \
          batch_size = BATCH_SIZE, epochs = EPOCHS, verbose=2, \
          validation_data=(valid_padded, valid.toxic.values), \
          callbacks=[modelcheckpoint])
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
path_to_glove_file = "../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))
word_index = tokenizer.word_index
num_tokens = len(word_index) + 1
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
from keras import initializers

with strategy.scope():
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=100, \
                                      embeddings_initializer=initializers.Constant(embedding_matrix), \
                                      trainable=False),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(padded, train1.toxic.values, epochs = 10, validation_data=(valid_padded, valid.toxic.values), verbose=2)
#0.5760
with strategy.scope():
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=100, \
                                      embeddings_initializer=initializers.Constant(embedding_matrix), \
                                      trainable=False),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, train1.toxic.values, epochs = 5, validation_data=(valid_padded, valid.toxic.values), verbose=2)
# 0.8461
train.head()
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = []
    for text in texts:
        enc = tokenizer.encode(texts, 
                               return_attention_masks=False, 
                               return_token_type_ids=False,
                               pad_to_max_length=True,
                               max_length=maxlen)
        enc_di.append(enc)
    
    return np.array(enc_di)
tokenizer.encode(train.comment_text[0], 
                               return_attention_masks=False, 
                               return_token_type_ids=False,
                               pad_to_max_length=True,
                               max_length=maxlen)
texts = train.comment_text
for text in texts:
    print(text)
[x for x in regular_encode(train.comment_text[0], tokenizer, 512)]
x_train = regular_encode(train.comment_text.values, tokenizer, 512)
x_valid = regular_encode(valid.comment_text.values, tokenizer, 512)
y_train = train.toxic.values
y_valid = valid.toxic.values
del train1_csv, train2_csv, train1, train2
gc.collect()
del train2_high, train2_low
gc.collect()
# prompt tf.data runtime to tune the prefetch value dynamically at runtime.
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
train_dataset = (tf.data.Dataset\
                .from_tensor_slices((x_train, y_train))\
                .shuffle(2048)\
                .batch(BATCH_SIZE)\
                .prefetch(AUTO))

valid_dataset = (tf.data.Dataset\
                .from_tensor_slices((x_valid, y_valid))\
                .batch(BATCH_SIZE)\
                .cache()\
                .prefetch(AUTO))
def build_model(transformer, max_len=512):
    inp = Input(shape=(max_len, ), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(inp)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model
with strategy.scope():
    transformer_layer = TFXLMRobertaModel.from_pretrained('jplu/tf-xlm-roberta-large')
    model_f = build_model(transformer_layer, max_len=MAX_LEN)
model_f.summary()
n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model_f.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
test.head()
sub = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sub.head()
x_test = regular_encode(test.content.values, tokenizer)
test_dataset = (tf.data.Dataset\
                .from_tensor_slices(x_test)\
                .batch(BATCH_SIZE))
sub['toxic'] = model_f.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)
