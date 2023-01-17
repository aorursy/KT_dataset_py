import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time

%matplotlib inline
df = pd.read_csv('../input/enron-email-dataset/emails.csv', nrows=500)

df.info()
print(df.file[2])
example = df.message[2]

print(example)
email_lines = ['Message-ID', 'Date', 'From', 'To', 'Subject', 'Cc', 'Mime-Version',\

               'Content-Type', 'Content-Transfer-Encoding', 'Bcc', 'X-From', 'X-To',\

               'X-cc', 'X-bcc', 'X-Folder', 'X-Origin', 'X-FileName', 'Content']
def get_content(message):

    message = message.split('\n')

    index = message.index('')

    message = message[: index] + ['Content: ' + ''.join(message[index+1:])]

    message = [message[i].split(': ', 1) for i in range(len(message))]

    i = 0

    while i < len(message):

        if len(message[i]) != 2:

            message[i-1][1] += message[i][0]

            message.pop(i)

        elif message[i][0] not in email_lines:

            message[i-1][1] += (message[i][0] + ': ' + message[i][1])

            message.pop(i)

        else:

            i += 1

    return message
get_content(example)
list_of_content = []

for i in range(500):

    list_of_content += [pd.DataFrame(get_content(df.message[i])).set_index(0).transpose()]
data = pd.concat(list_of_content, axis=0, sort=False).reset_index()

data['Index'] = df['file']

col = ['Index'] + [i[0] for i in get_content(df.message[86])]

data = data[col].replace('', np.nan)

print(data.shape)

data.head()
data.dtypes
analysis_data = data[['Date', 'From', 'To', 'Subject', 'Content']].dropna().copy()

analysis_data['Date'] =  pd.to_datetime(analysis_data['Date'])

print(analysis_data.shape)

analysis_data.head()
beeap = pd.read_csv('../input/label-beeap/BEEAP.csv')

beeap.drop('Unnamed: 0', axis=1, inplace=True)

for i in beeap.columns[1:]:

    beeap[i] = beeap[i].astype(np.float64, errors='ignore')

print(beeap.shape)

beeap.head()
plt.figure(figsize=(10, 6))

plt.plot(sorted(beeap.Content.str.len()))

plt.title('Distribution of Content Length', fontsize=16)

plt.xlim(-50, 1750)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.grid()

plt.show()
beeap_1 = pd.read_csv('../input/beeapfinal/beeap_1.csv')

beeap_1.drop('Unnamed: 0', axis=1, inplace=True)

for i in beeap_1.columns[1:]:

    beeap_1[i] = beeap_1[i].astype(np.float64, errors='ignore')

print(beeap_1.shape)

beeap_1.head()
# https://deeplearningcourses.com/c/deep-learning-advanced-nlp

from __future__ import print_function, division

from builtins import range

# Note: you may need to update your version of future

# sudo pip install -U future



import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer

# we want our data have the same length

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from sklearn.metrics import roc_auc_score



# some configuration

# can use the maximum of the sequence in one email and set it larger

max_len = 200

# a native english user knows 20000 words in practice...

max_features = 100000

# the size of each word vector ... using the pretrained model

embed_size = 300

VALIDATION_SPLIT = 0.2

batch_size = 128

epochs = 50



# Download the data:

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# Download the word vectors:

# http://nlp.stanford.edu/data/glove.6B.zip



print('Loading word vectors...')

word2vec = {}

#with open(os.path.join('../input/glove840b300dtxt/glove.840B.%sd.txt' % embed_size)) as f:

with open(os.path.join('../input/glove-6b/glove.6B.%sd.txt' % embed_size)) as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

    for line in f:

        values = line.split()

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32')

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
# prepare text samples and their labels

print('Loading in comments...')

train = beeap_1

# extract the comments, fill NaN with some values

sentences = train["Contents"].fillna("DUMMY_VALUE").values

# possible_labels_details = ["Business", "Personal", "Personal but professional", "Logistic", "Employment", "Document", 'Empty attachment', 'Empty']

possible_labels= [str(i+1) for i in range(6)]

# possible_labels= [i+1 for i in range(13)]

targets = train[possible_labels].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(sentences, targets, test_size = 0.2)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125)
# convert the sentences (strings) into integers， thus they can be used as index later on

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)

#sequences = tokenizer.texts_to_sequences(sentences)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_valid_seq = tokenizer.texts_to_sequences(X_valid)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# print("sequences:", sequences); exit()





print("max sequence length:", max(len(s) for s in X_train_seq))

print("min sequence length:", min(len(s) for s in X_train_seq))

s = sorted(len(s) for s in X_train_seq)

print("median sequence length:", s[len(s) // 2])





# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))





# pad sequences so that we get a N x T matrix

# Keras take care of the 0 only for padding purpose 

#data = pad_sequences(sequences, maxlen=max_len)

X_train = pad_sequences(X_train_seq, maxlen=max_len)

X_valid = pad_sequences(X_valid_seq, maxlen=max_len)

X_test = pad_sequences(X_test_seq, maxlen=max_len)

print('Shape of data tensor:', X_train.shape)







# prepare embedding matrix

print('Filling pre-trained embeddings...')

num_words = min(max_features, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word2idx.items():

    if i < max_features:

        embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

        embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(

  min(max_features, embedding_matrix.shape[0]),

  embed_size,

  weights=[embedding_matrix],

  input_length=max_len,

    # don't want to make the embeddding updated during the procedure

  trainable=False

)
import logging

from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback



class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D



print('Building model...')



file_path = "best_model.cnn"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)



def build_cnn_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    input_ = Input(shape=(max_len,))

    x = embedding_layer(input_)

    x = Conv1D(128, 3, activation='relu')(x)

    x = MaxPooling1D(3)(x)

    x = Conv1D(128, 3, activation='relu')(x)

    x = MaxPooling1D(3)(x)

    x = Conv1D(128, 3, activation='relu')(x)

    x = GlobalMaxPooling1D()(x)

    x = Dense(128, activation='relu')(x)

    # using sigmoid since we are doing six binary classifications

    output = Dense(len(possible_labels), activation='sigmoid')(x)

    

    model = Model(inputs = input_, outputs = output)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_valid, y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_cnn_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
print('Building model...')



file_path = "best_model.lstm"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)



def build_lstm_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape=(max_len,))

    x = embedding_layer(inp)

    x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    # using sigmoid since we are doing six binary classifications

    output = Dense(len(possible_labels), activation='sigmoid')(x)

    

    model = Model(inputs = inp, outputs = output)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_valid, y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_lstm_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
print('Building model...')



file_path = "best_model.bilstm"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)



def build_bilstm_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape = (max_len,))

    x = embedding_layer(inp)

    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

    x = GlobalMaxPool1D()(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_valid, y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_bilstm_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
print('Building model...')



file_path = "best_model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)



def build_hdf5_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape = (max_len,))

    x = embedding_layer(inp)

    x = SpatialDropout1D(dr)(x)



    x = Bidirectional(GRU(units, return_sequences = True))(x)

    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, max_pool])



    x = Dense(6, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_valid, y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_hdf5_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
print('Building model...')



file_path = "best_model.advanced"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)



def build_advanced_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape=(max_len,))

    x = embedding_layer(inp)

    x1 = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(units, return_sequences = True))(x1)

    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)

    y = Bidirectional(LSTM(units, return_sequences = True))(x1)

    y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    avg_pool2 = GlobalAveragePooling1D()(y)

    max_pool2 = GlobalMaxPooling1D()(y)

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    output = Dense(len(possible_labels), activation='sigmoid')(x)

    

    model = Model(inputs = inp, outputs = output)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_valid, y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_advanced_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
from sklearn.metrics import roc_curve, auc

from scipy import interp



fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(6):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# Aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))

# Interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(6):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC

mean_tpr /= 6



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure(figsize=(10, 10))

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='gold', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=2)



for i in range(6):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC Curves for Coarse Genre', fontsize=16)

plt.legend()

plt.show()
def pred_classes(y_pred):

    train_pred = model.predict(X_train, batch_size = batch_size, verbose = 1)

    threshold = []

    for x in range(6):

        result = []

        for i in np.arange(0, 1.01, 0.001):

            result += [sum(y_train[:, x] != (train_pred[:, x] >= i))]

        result = np.array(result)

        threshold += [round(np.where(result == result.min())[0].mean())/1000]

    pred_classes = y_pred >= threshold

    return pred_classes.astype(int)
y_pred = pred_classes(pred)

for i in range(6):

    print('Accuracy for class ' + str(i) + ': ' + str(round(sum(y_pred[:, i] == y_test[:, i])/y_pred.shape[0], 4)))
### Try to write pred_classes with scipy minimize

#from scipy.optimize import minimize

#

#def function(threshold):

#    return sum(y_test[:, 0] != (pred[:, 0] >= threshold))



#threshold = np.random.rand()

#print(threshold)

#minimize(function, threshold)
from keras_self_attention import SeqSelfAttention



print('Building model...')



file_path = "best_model.attn"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)



def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape=(max_len,))

    x = embedding_layer(inp)

    x = Bidirectional(LSTM(units, return_sequences = True))(x)

    x = SeqSelfAttention(attention_activation='sigmoid')(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    # using sigmoid since we are doing six binary classifications

    output = Dense(len(possible_labels), activation='sigmoid')(x)

    

    model = Model(inputs = inp, outputs = output)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_valid, y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
#model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

#pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

#roc_auc_score(y_test, pred)
# some configuration

# can use the maximum of the sequence in one email and set it larger

max_len = 200

# a native english user knows 20000 words in practice...

max_features = 100000

# the size of each word vector ... using the pretrained model

embed_size = 300

VALIDATION_SPLIT = 0.2

batch_size = 128

epochs = 50



# Download the data:

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# Download the word vectors:

# http://nlp.stanford.edu/data/glove.6B.zip



print('Loading word vectors...')

word2vec = {}

#with open(os.path.join('../input/glove840b300dtxt/glove.840B.%sd.txt' % embed_size)) as f:

with open(os.path.join('../input/word2vec-ec/word2vec_ec/Word2Vec_ec.%sB.txt' % embed_size)) as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

    for line in f:

        values = line.split()

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32')

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
# prepare text samples and their labels

print('Loading in comments...')

train = beeap_1

# extract the comments, fill NaN with some values

sentences = train["Contents"].fillna("DUMMY_VALUE").values

# possible_labels_details = ["Business", "Personal", "Personal but professional", "Logistic", "Employment", "Document", 'Empty attachment', 'Empty']

possible_labels= [str(i+1) for i in range(6)]

# possible_labels= [i+1 for i in range(13)]

targets = train[possible_labels].values
X_train, X_test, y_train, y_test = train_test_split(sentences, targets, test_size = 0.2)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125)
# convert the sentences (strings) into integers， thus they can be used as index later on

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)

#sequences = tokenizer.texts_to_sequences(sentences)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_valid_seq = tokenizer.texts_to_sequences(X_valid)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# print("sequences:", sequences); exit()





print("max sequence length:", max(len(s) for s in X_train_seq))

print("min sequence length:", min(len(s) for s in X_train_seq))

s = sorted(len(s) for s in X_train_seq)

print("median sequence length:", s[len(s) // 2])





# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))





# pad sequences so that we get a N x T matrix

# Keras take care of the 0 only for padding purpose 

#data = pad_sequences(sequences, maxlen=max_len)

X_train = pad_sequences(X_train_seq, maxlen=max_len)

X_valid = pad_sequences(X_valid_seq, maxlen=max_len)

X_test = pad_sequences(X_test_seq, maxlen=max_len)

print('Shape of data tensor:', X_train.shape)







# prepare embedding matrix

print('Filling pre-trained embeddings...')

num_words = min(max_features, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word2idx.items():

    if i < max_features:

        embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

        embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(

  min(max_features, embedding_matrix.shape[0]),

  embed_size,

  weights=[embedding_matrix],

  input_length=max_len,

    # don't want to make the embeddding updated during the procedure

  trainable=False

)
model = build_advanced_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(6):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# Aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))

# Interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(6):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC

mean_tpr /= 6



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure(figsize=(10, 10))

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='gold', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=2)



for i in range(6):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC Curves for Coarse Genre', fontsize=16)

plt.legend()

plt.show()
# some configuration

# can use the maximum of the sequence in one email and set it larger

max_len = 200

# a native english user knows 20000 words in practice...

max_features = 100000

# the size of each word vector ... using the pretrained model

embed_size = 300

VALIDATION_SPLIT = 0.2

batch_size = 128

epochs = 50



# Download the data:

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# Download the word vectors:

# http://nlp.stanford.edu/data/glove.6B.zip



print('Loading word vectors...')

word2vec = {}

#with open(os.path.join('../input/glove840b300dtxt/glove.840B.%sd.txt' % embed_size)) as f:

with open(os.path.join('../input/glove-ec/glove_ec/GloVe_ec.%sB.txt' % embed_size)) as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

    for line in f:

        values = line.split()

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32')

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
# prepare text samples and their labels

print('Loading in comments...')

train = beeap_1

# extract the comments, fill NaN with some values

sentences = train["Content"].fillna("DUMMY_VALUE").values

# possible_labels_details = ["Business", "Personal", "Personal but professional", "Logistic", "Employment", "Document", 'Empty attachment', 'Empty']

possible_labels= [str(i+1) for i in range(6)]

# possible_labels= [i+1 for i in range(13)]

targets = train[possible_labels].values
X_train, X_test, y_train, y_test = train_test_split(sentences, targets, test_size = 0.2)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125)
# convert the sentences (strings) into integers， thus they can be used as index later on

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_valid_seq = tokenizer.texts_to_sequences(X_valid)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# print("sequences:", sequences); exit()





print("max sequence length:", max(len(s) for s in sequences))

print("min sequence length:", min(len(s) for s in sequences))

s = sorted(len(s) for s in sequences)

print("median sequence length:", s[len(s) // 2])





# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))





# pad sequences so that we get a N x T matrix

# Keras take care of the 0 only for padding purpose 

data = pad_sequences(sequences, maxlen=max_len)

X_train = pad_sequences(X_train_seq, maxlen=max_len)

X_valid = pad_sequences(X_valid_seq, maxlen=max_len)

X_test = pad_sequences(X_test_seq, maxlen=max_len)

print('Shape of data tensor:', data.shape)







# prepare embedding matrix

print('Filling pre-trained embeddings...')

num_words = min(max_features, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word2idx.items():

    if i < max_features:

        embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

        embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(

  min(max_features, embedding_matrix.shape[0]),

  embed_size,

  weights=[embedding_matrix],

  input_length=max_len,

    # don't want to make the embeddding updated during the procedure

  trainable=False

)
model = build_advanced_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(6):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# Aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))

# Interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(6):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC

mean_tpr /= 6



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure(figsize=(10, 10))

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='gold', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=2)



for i in range(6):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC Curves for Coarse Genre', fontsize=16)

plt.legend()

plt.show()
model.summary()
beeap_2 = pd.read_csv('../input/beeapfinal/beeap_2.csv')

beeap_2.drop('Unnamed: 0', axis=1, inplace=True)

for i in beeap_2.columns[1:]:

    beeap_2[i] = beeap_2[i].astype(np.float64, errors='ignore')

print(beeap_2.shape)

beeap_2.head()
# prepare text samples and their labels

print('Loading in comments...')

train = beeap_2

# extract the comments, fill NaN with some values

sentences = train["Contents"].fillna("DUMMY_VALUE").values

# possible_labels_details = ["Business", "Personal", "Personal but professional", "Logistic", "Employment", "Document", 'Empty attachment', 'Empty']

possible_labels= [str(i+1) for i in range(13)]

# possible_labels= [i+1 for i in range(13)]

targets = train[possible_labels].values
X_train, X_test, y_train, y_test = train_test_split(sentences, targets, test_size = 0.2, random_state=8)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state=8)
# convert the sentences (strings) into integers， thus they can be used as index later on

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)

#sequences = tokenizer.texts_to_sequences(sentences)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_valid_seq = tokenizer.texts_to_sequences(X_valid)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# print("sequences:", sequences); exit()





print("max sequence length:", max(len(s) for s in X_train_seq))

print("min sequence length:", min(len(s) for s in X_train_seq))

s = sorted(len(s) for s in X_train_seq)

print("median sequence length:", s[len(s) // 2])





# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))





# pad sequences so that we get a N x T matrix

# Keras take care of the 0 only for padding purpose 

#data = pad_sequences(sequences, maxlen=max_len)

X_train = pad_sequences(X_train_seq, maxlen=max_len)

X_valid = pad_sequences(X_valid_seq, maxlen=max_len)

X_test = pad_sequences(X_test_seq, maxlen=max_len)

print('Shape of data tensor:', X_train.shape)







# prepare embedding matrix

print('Filling pre-trained embeddings...')

num_words = min(max_features, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word2idx.items():

    if i < max_features:

        embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

        embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(

  min(max_features, embedding_matrix.shape[0]),

  embed_size,

  weights=[embedding_matrix],

  input_length=max_len,

    # don't want to make the embeddding updated during the procedure

  trainable=False

)
ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)



model = build_advanced_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(13):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# Aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(13)]))

# Interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(13):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC

mean_tpr /= 13



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure(figsize=(10, 10))

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='gold', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=2)



for i in range(13):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC Curves for Coarse Genre', fontsize=16)

plt.legend()

plt.show()
beeap_3 = pd.read_csv('../input/beeapfinal/beeap_3.csv')

beeap_3.drop('Unnamed: 0', axis=1, inplace=True)

for i in beeap_3.columns[1:]:

    beeap_3[i] = beeap_3[i].astype(np.float64, errors='ignore')

print(beeap_3.shape)

beeap_3.head()
# prepare text samples and their labels

print('Loading in comments...')

train = beeap_2

# extract the comments, fill NaN with some values

sentences = train["Contents"].fillna("DUMMY_VALUE").values

# possible_labels_details = ["Business", "Personal", "Personal but professional", "Logistic", "Employment", "Document", 'Empty attachment', 'Empty']

possible_labels= [str(i+1) for i in range(13)]

# possible_labels= [i+1 for i in range(13)]

targets = train[possible_labels].values
X_train, X_test, y_train, y_test = train_test_split(sentences, targets, test_size = 0.2, random_state=8)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state=8)
# convert the sentences (strings) into integers， thus they can be used as index later on

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(sentences)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_valid_seq = tokenizer.texts_to_sequences(X_valid)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# print("sequences:", sequences); exit()





print("max sequence length:", max(len(s) for s in sequences))

print("min sequence length:", min(len(s) for s in sequences))

s = sorted(len(s) for s in sequences)

print("median sequence length:", s[len(s) // 2])





# get word -> integer mapping

word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))





# pad sequences so that we get a N x T matrix

# Keras take care of the 0 only for padding purpose 

data = pad_sequences(sequences, maxlen=max_len)

X_train = pad_sequences(X_train_seq, maxlen=max_len)

X_valid = pad_sequences(X_valid_seq, maxlen=max_len)

X_test = pad_sequences(X_test_seq, maxlen=max_len)

print('Shape of data tensor:', data.shape)







# prepare embedding matrix

print('Filling pre-trained embeddings...')

num_words = min(max_features, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word2idx.items():

    if i < max_features:

        embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

        embedding_matrix[i] = embedding_vector
# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(

  min(max_features, embedding_matrix.shape[0]),

  embed_size,

  weights=[embedding_matrix],

  input_length=max_len,

    # don't want to make the embeddding updated during the procedure

  trainable=False

)
ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)



model = build_advanced_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

pred = model.predict(X_test, batch_size = batch_size, verbose = 1)
# plot the mean AUC over each label

roc_auc_score(y_test, pred)
fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(13):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# Compute macro-average ROC curve and ROC area

# Aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(13)]))

# Interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(13):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC

mean_tpr /= 13



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure(figsize=(10, 10))

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='gold', linestyle=':', linewidth=2)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=2)



for i in range(13):

    plt.plot(fpr[i], tpr[i],

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC Curves for Coarse Genre', fontsize=16)

plt.legend()

plt.show()