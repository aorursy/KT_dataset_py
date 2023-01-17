import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import os

from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU

from keras.callbacks import Callback

from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten

from keras.preprocessing import text, sequence

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Model

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.utils import plot_model

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from sklearn.metrics import precision_recall_fscore_support



print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/jigsawtraintest/train_jigsaw.csv")

test= pd.read_csv("../input/jigsawtraintest/test_jigsaw.csv")

EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
train["comment_text"].fillna("fillna")

test["comment_text"].fillna("fillna")

X_train = train["comment_text"].str.lower()

y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values



X_test = test["comment_text"].str.lower()
max_features=100000

maxlen=150

embed_size=300
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
tok=text.Tokenizer(num_words=max_features,lower=True)

tok.fit_on_texts(list(X_train)+list(X_test))

X_train=tok.texts_to_sequences(X_train)

X_test=tok.texts_to_sequences(X_test)

x_train=sequence.pad_sequences(X_train,maxlen=maxlen)

x_test=sequence.pad_sequences(X_test,maxlen=maxlen)
embeddings_index = {}

with open(EMBEDDING_FILE,encoding='utf8') as f:

    for line in f:

        values = line.rstrip().rsplit(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs
word_index = tok.word_index

#prepare embedding matrix

num_words = min(max_features, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embed_size))

for word, i in word_index.items():

    if i >= max_features:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
sequence_input = Input(shape=(maxlen, ))

x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)

x = SpatialDropout1D(0.2)(x)

x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)

x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

x = concatenate([avg_pool, max_pool]) 

# x = Dense(128, activation='relu')(x)

# x = Dropout(0.1)(x)

preds = Dense(6, activation="sigmoid")(x)

model = Model(sequence_input, preds)

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])
model.summary()
from keras.utils.vis_utils import plot_model
SVG(model_to_dot(model).create(prog='dot', format='svg'))
import keras

keras.utils.plot_model(model, 'my_first_model.png')
batch_size = 128

epochs = 5

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
filepath="weights_base.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_acc", mode="max", patience=5)

ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)

callbacks_list = [ra_val,checkpoint, early]
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)

#Loading model weights

model.load_weights(filepath)

print('Predicting....')

y_pred = model.predict(x_test,batch_size=1024,verbose=1)
y_df = np.where(y_pred > 0.5, 1, 0)
y_df = pd.DataFrame(y_df, columns=['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

y_df = y_df.astype('int')
labels = ['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
def get_metri_scores(y_test, y_test_pred):

    vals = precision_recall_fscore_support(y_test, y_test_pred, average='macro')

    precision = vals[0]

    recall = vals[1]

    f1 = vals[2]

    acc = accuracy_score(y_test, y_test_pred)

    return precision, recall, f1, acc
results_cv = pd.DataFrame({'labels': labels})

results_cv['acc'] = 0

results_cv['f1'] = 0

results_cv['precision'] = 0

results_cv['recall']  = 0

for col in labels:

    print(col)

    precision, recall, f1, acc = get_metri_scores(test[col], y_df[col])

    results_cv['acc'][results_cv['labels']==col] = acc

    results_cv['f1'][results_cv['labels']==col] = f1

    results_cv['precision'][results_cv['labels']==col] = precision

    results_cv['recall'][results_cv['labels']==col] = recall
results_cv
#Base Model
sequence_input = Input(shape=(maxlen, ))

x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)

x = SpatialDropout1D(0.2)(x)

x = LSTM(256, return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(x)

x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)

max_pool = GlobalMaxPooling1D()(x)

x = Dense(128, activation='relu')(max_pool)

x = Dropout(0.1)(x)

preds = Dense(6, activation="sigmoid")(x)

model2 = Model(sequence_input, preds)

model2.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])
SVG(model_to_dot(model2).create(prog='dot', format='svg'))
filepath="weights_novice_model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_acc", mode="max", patience=5)

ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)

callbacks_list = [ra_val,checkpoint, early]



model2.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)

#Loading model weights

model2.load_weights(filepath)

print('Predicting....')

y_pred = model2.predict(x_test,batch_size=1024,verbose=1)
y_df = np.where(y_pred > 0.5, 1, 0)

y_df = pd.DataFrame(y_df, columns=['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

y_df = y_df.astype('int')
def get_metri_scores(y_test, y_test_pred):

    vals = precision_recall_fscore_support(y_test, y_test_pred, average='macro')

    precision = vals[0]

    recall = vals[1]

    f1 = vals[2]

    acc = accuracy_score(y_test, y_test_pred)

    return precision, recall, f1, acc







results_cv = pd.DataFrame({'labels': labels})

results_cv['acc'] = 0

results_cv['f1'] = 0

results_cv['precision'] = 0

results_cv['recall']  = 0

for col in labels:

    print(col)

    precision, recall, f1, acc = get_metri_scores(test[col], y_df[col])

    results_cv['acc'][results_cv['labels']==col] = acc

    results_cv['f1'][results_cv['labels']==col] = f1

    results_cv['precision'][results_cv['labels']==col] = precision

    results_cv['recall'][results_cv['labels']==col] = recall
results_cv