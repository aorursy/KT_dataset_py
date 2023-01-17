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
import numpy as np
np.random.seed(233)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re, gc
from keras import optimizers, utils
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, MaxPool2D
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, MaxPooling1D, BatchNormalization, Reshape
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
df_train = pd.read_csv('../input/moviereviewsentimentalanalysis/train.tsv', sep='\t', header=0)
feature_names = list(df_train.columns.values)
X_train = df_train['Phrase'].values
Y_train = df_train['Sentiment'].values

df_test = pd.read_csv('../input/moviereviewsentimentalanalysis/test.tsv', sep='\t', header=0)
X_test = df_test['Phrase'].values
X_test_PhraseID = df_test['PhraseId']
max_features = 20000
maxlen = 160
embed_size = 300

def pad(text):
    if (len(text) < 90):
        text += " "
    while (len(text) < 90):
        text += text 
    return text

X_train = df_train['Phrase'].apply(pad).values
X_test = df_test['Phrase'].apply(pad).values

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

Y_train = utils.to_categorical(Y_train, 5)
word_index = tokenizer.word_index
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
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
print ("begin trainnig")
#写一个LossHistory类，保存loss和acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
def TextCNN_1():
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    conv1 = MaxPooling1D(maxlen-2)(conv1)
    conv1 = Flatten()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    conv2 = MaxPooling1D(maxlen-1)(conv2)
    conv2 = Flatten()(conv2)
    conv3 = Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')(x)
    conv3 = MaxPooling1D(maxlen-3)(conv3)
    conv3 = Flatten()(conv3)
    
    # x = Bidirectional(GRU(60, return_sequences=True))(x)
    # x = GlobalMaxPooling1D()(x)
    # x = Dropout(0.1)(x)
    # x = Dense(40, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([conv1, conv2, conv3])
    outp = Dense(5, activation="softmax")(conc)
    
    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


filter_sizes = [2,3,4]
num_filters = 32

def TextCNN():    
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                    activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                    activation='relu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                    activation='relu')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    x1 = Flatten()(maxpool_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    x2 = Flatten()(maxpool_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    x3 = Flatten()(maxpool_2)
    
    conc = concatenate([x1, x2, x3])
    outp = Dense(5, activation="softmax")(conc)
    
    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])

    return model


def BiGRU():
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(5, activation="softmax")(conc)
    
    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['accuracy'])

    return model
batch_size = 256
epochs = 4
model = TextCNN()
model.summary()
[X_tra, X_val, y_tra, y_val] = train_test_split(X_train, Y_train, train_size=0.95, random_state=233)

history = LossHistory()
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[earlyStopping, history], verbose=1)
scores = model.evaluate(X_val, y_val, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
from keras.utils import plot_model
plot_model(model, to_file='model.png')
import matplotlib.pyplot as plt
%matplotlib inline
history.loss_plot('epoch')
history.loss_plot('batch')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
f = open('./Submission.csv', 'w')
f.write('PhraseId,Sentiment\n')


predicted_classes = model.predict(X_test, batch_size=512, verbose=1)
predicted_classes = np.argmax(predicted_classes, axis=1)
for i in range(0,X_test_PhraseID.shape[0]):
    f.write(str(X_test_PhraseID[i])+","+str(predicted_classes[i])+'\n')

f.close()
sentence = "this movie is good"
sentence = pad(sentence)
sentence = tokenizer.texts_to_sequences([sentence])
sentence = sequence.pad_sequences(sentence, maxlen=maxlen)
prediction = model.predict(sentence, batch_size=1, verbose=1)
prediction = np.argmax(prediction, axis=1)
print (prediction)
num = 9999
text = df_train['Phrase'].values[num]
label = df_train['Sentiment'].values[num]
print("Raw text: ", text)
text = pad(text)
sentence = tokenizer.texts_to_sequences([text])
sentence = sequence.pad_sequences(sentence, maxlen=maxlen)
prediction = model.predict(sentence, batch_size=1, verbose=1)
prediction = np.argmax(prediction, axis=1)
print ("prediction: :",prediction)
print("True label: ", label)


print("An")
print(pad("An"))
