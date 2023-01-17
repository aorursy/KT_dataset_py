# import some libaries
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, LSTM, GRU, BatchNormalization
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, Flatten, Bidirectional
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
# What data we have 
print(os.listdir("../input/"))
# load train and test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train datasets shape:", train.shape)
print("Test datasets shape:", test.shape)
# Show some train datasets 
print('Train data samples:')
train.head()
print('Test data samples')
test.head()
# What ratio for insincere data
# is_not_in_ratio = train.target.value_counts()[0]/len(train)
# is_in_ratio = train.target.value_counts()[1]/len(train)

# How many different parts numbers
sns.countplot(train.target)
plt.show()
# This function is used to get some basic information(how many words and characters) about this text
def cfind(df):
    df_new = df.copy()
    data = df.question_text
    df_new['Sentence_length'] = pd.Series([len(r) for r in data])
    df_new['Word_num'] = pd.Series([len(r.split(' ')) for r in data])
    return df_new
train_new = cfind(train)
test_new = cfind(test)
# Plot the basic information
fig, ax = plt.subplots(1, 2, figsize=(14, 10))
sns.distplot(train_new.Sentence_length, ax=ax[0])
ax[0].set_title('Sentence Length distribution')
sns.distplot(train_new.Word_num, ax=ax[1])
ax[1].set_title('Word number distribution')
plt.legend()
plt.show()
# Here I will split the data to train and validation data
train_data, validation_data = train_test_split(train_new, test_size=.1, random_state=1234)
# Here I will use Tokenizer to extract the keyword vector as baseline
# I will use train data to fit the Tokenizer, then use this Tokenizer to extract the validation data
# max_length = 100
# max_features = 50000
# token = Tokenizer(num_words=max_features)
# token.fit_on_texts(list(np.asarray(train_data.question_text)))
# xtrain = token.texts_to_sequences(np.asarray(train_data.question_text))
# xvalidate = token.texts_to_sequences(np.asarray(validation_data.question_text))
# xtest = token.texts_to_sequences(np.asarray(test_new.question_text))

# # Because Tokenizer will split the sentence, for some sentence are smaller,
# # so we have to pad the missing position
# xtrain = pad_sequences(xtrain, maxlen=max_length)
# xvalidate = pad_sequences(xvalidate, maxlen=max_length)
# xtest = pad_sequences(xtest, maxlen=max_length)

# ytrain = train_data.target
# yvaliate = validation_data.target
# Here I write a helper function to evaluate model
def evaluate(y, pred, p_diff_thre=False):
    if pred.shape[1] == 2:
        pred = np.argmax(pred, axis=1)
    if y.shape[1] == 2:
        y = np.argmax(y, axis=1)
    score = metrics.f1_score(y, pred)
    print('F1-score=%.4f'%(score))
    if p_diff_thre:
        for thr in np.arange(.1, .501, .01):
            thr = np.round(thr, 2)
            print('Threshold: %.3f, F1-Score: %.4f'%(thr, metrics.f1_score(y, (pred>thr).astype(int))))
    return score

### Because of training this LR model is so slow, not train it for now.
# # Here is a baseline model: Logistic Regression
# from sklearn.linear_model import LogisticRegression
# lr_base = LogisticRegression()
# lr_base.fit(xtrain, ytrain)
# pred_lr_base = lr_base.predict(xvalidate)
# evaluate(yvaliate, pred_lr_base)
# Before I use Doc2Vec algorithm, I will use nltk to do word tokenizer.
# First must Tagged each document with index
tag_d = np.array(train_new.question_text)
tagged_data = [TaggedDocument(words=word_tokenize(d.lower()), 
                              tags=[str(i)]) for i, d in enumerate(tag_d)]
### Noted: to train Doc2vec model, there are some important parameters, like: alpha(learning rate), 
### vector_size(how many dimension of result vector), dm(whether to use distributed bags of words)
### So if decided to use Doc2Vec model, have to tune these parameters to get a better representation.
# After I have get tagged data, then I will start to train Doc2vec model
def doc2vec_training(tagged_data):
    epochs = 10   # How many epochs to be trained
    vector_size = 256  # How many dimensions
    alpha = .025   # Initial learning rate
    min_alpha = .00025  # Learning rate changes step
    dm = 1   # Use Distributed bags of words
    
    # Start build model
    model = Doc2Vec(vector_size=vector_size, alpha=alpha, min_alpha=min_alpha, min_count=1, dm=dm, workers=-1)
    model.build_vocab(tagged_data)
    
    # start to train model
    for e in range(epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        # decrease learning rate
        model.alpha -= min_alpha
        model.min_alpha = model.alpha
        print('Now is epochs: %02d'%(e))
    print('Finished model training process')
    
    return model

# Because I want to get the return result data vector, 
# but I will also want to use this model to infer test data, so this is return model
doc2vec_model = doc2vec_training(tagged_data)
# After I have trained this model, I will use this model to infer train datasets
# For now, I will not infer the test datasets, because I don't know whether this feature is best to test.
# doc2vec_train = doc2vec_model.infer_vector(tag_d)
# Get final result from already trained doc2vec model.
doc2vec_train = doc2vec_model.docvecs.vectors_docs
# After I get the doc2vec result, then I will split this result to train and valiatation result.
from tensorflow import keras
label = np.asarray(train_new.target).reshape(-1, 1)
label = keras.utils.to_categorical(label)
xtrain_doc, xtest_doc, ytrain_doc, ytest_doc = train_test_split(doc2vec_train, label, test_size=.2, random_state=1234)
# Here I will build a DNN model as deep learning baseline

# Here I write a DNN class for many other cases, 
# you can choose how many layers, how many units, whether to use dropout,
# whether to use batchnormalization, also with optimizer! 
class dnnNet(object):
    def __init__(self, n_classes=2, n_dims=None, n_layers=3, n_units=64, use_dropout=True, drop_ratio=.5, use_batchnorm=True,
                 metrics='accuracy', optimizer='rmsprop'):
        self.n_classes = n_classes
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_units = n_units
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batchnorm = use_batchnorm
        self.metrics = metrics
        self.optimizer = optimizer
        self.model = self._init_model()

    def _init_model(self):
        if self.n_dims is None:
            raise AttributeError('Data Dimension must be provided!')
        inputs = Input(shape=(self.n_dims, ))

        # this is dense block function.
        def _dense_block(layers):
            res = Dense(self.n_units)(layers)
            if self.use_batchnorm:
                res = BatchNormalization()(res)
            res = Activation('relu')(res)
            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)
            return res

        for i in range(self.n_layers):
            if i == 0:
                res = _dense_block(inputs)
            else: res = _dense_block(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameters n_class must be provide up or equal 2!')

        return model

    # For fit function, auto randomly split the data to be train and validation datasets.
    def fit(self, data, label, epochs=100, batch_size=256):
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1,
                                  validation_data=(xvalidate, yvalidate))
        print('Model evaluation on validation datasets accuracy:{:.4f}'.format(self.model.evaluate(xvalidate, yvalidate)[1]))
        return self

    def evaluate(self, data, label, batch_size=None, silent=False):
        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.6f}'.format(acc))
        return acc
    
    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)
    
    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.plot(self.his.history['acc'], label='Train Accuracy')
        ax1.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Train and Validation Accuracy Curve')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.plot(self.his.history['loss'], label='Train Loss')
        ax2.plot(self.his.history['val_loss'], label='Validation Loss')
        ax2.set_title('Train and Validation Loss Curve')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss score')
        plt.legend()
        plt.show()    
model_dnn = dnnNet(n_classes=2, n_dims=xtrain_doc.shape[1], n_layers=3, n_units=128)
model_dnn.fit(xtrain_doc, ytrain_doc, epochs=10, batch_size=2048)
acc = model_dnn.evaluate(xtest_doc, ytest_doc, batch_size=10240)
model_dnn.plot_acc_curve()
# Before we move forward, del some unused datasets
import gc
del train_new,train, doc2vec_model, doc2vec_train
gc.collect()
# Because for CNN, LSTM, Residual network and DenseNet, data must up to 2 dimensions,I convert data is 3-D(batch, 16, 16)
xtrain_deep = xtrain_doc.reshape(-1, 16, 16)
xtest_deep = xtest_doc.reshape(-1, 16, 16)
ytrain_deep = ytrain_doc.copy()
ytest_deep = ytest_doc.copy()
class residualNet(object):
    def __init__(self, input_dim1=None, input_dim2=None, n_classes=2, n_layers=4, flatten=True, use_dense=True,
                 n_dense_layers=1, conv_units=64, stride=1, padding='SAME', dense_units=128, drop_ratio=.5,
                 optimizer='rmsprop', metrics='accuracy'):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.flatten = flatten
        self.use_dense = use_dense
        self.n_dense_layers = n_dense_layers
        self.conv_units = conv_units
        self.stride = stride
        self.padding = padding
        self.dense_units = dense_units
        self.drop_ratio = drop_ratio
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = self._init_model()

    def _init_model(self):
        inputs = Input(shape=(self.input_dim1, self.input_dim2))

        # dense net residual block
        def _res_block(layers):
            res = Conv1D(self.conv_units, self.stride, padding=self.padding)(layers)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            res = Conv1D(self.input_dim2, self.stride, padding=self.padding)(res)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            return keras.layers.add([layers, res])

        # construct residual block chain.
        for i in range(self.n_layers):
            if i == 0:
                res = _res_block(inputs)
            else:
                res = _res_block(res)

        # using flatten or global average pooling to process Convolution result
        if self.flatten:
            res = Flatten()(res)
        else:
            res = GlobalAveragePooling1D()(res)

        # whether or not use dense net, also with how many layers to use
        if self.use_dense:
            for j in range(self.n_dense_layers):
                res = Dense(self.dense_units)(res)
                res = BatchNormalization()(res)
                res = Activation('relu')(res)
                res = Dropout(self.drop_ratio)(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            print('Model structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameters n_classes must up to 2!')

        return model

    # Fit on given training data and label. Here I will auto random split the data to train and validation data,
    # for test datasets, I will just use it if model already trained then I will evaluate the model.
    def fit(self, data, label, epochs=100, batch_size=256):
        # label is not encoding as one-hot, use keras util to convert it to one-hot
        if len(label.shape) == 1:
            label = keras.utils.to_categorical(label, num_classes=len(np.unique(label)))

        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, verbose=1, epochs=epochs,
                                  validation_data=(xvalidate, yvalidate), batch_size=batch_size)
        print('After training, model accuracy on validation datasets is {:.2f}%'.format(
            self.model.evaluate(xvalidate, yvalidate)[1]*100))
        return self

    # this is evaluation function to evaluate already trained model.
    def evaluate(self, data, label, batch_size=None, silent=False):
        if len(label.shape) == 1:
            label = keras.utils.to_categorical(label, num_classes=len(np.unique(label)))

        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.2f}%'.format(acc*100))
        return acc
    
    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)
    
    # plot after training accuracy and loss curve.
    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['acc'], label='Train Accuracy')
        ax.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax.set_title('Train and Validation Accruacy Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['loss'], label='Traing Loss')
        ax.plot(self.his.history['val_loss'], label='Validation Loss')
        ax.set_title('Train and Validation Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss score')

        plt.legend()
        plt.show()

### You must have noticed that Residual network will be more deep but with less parameters. This is amazing of Residual net.
# Start to build a residual network
model_res = residualNet(n_classes=2, input_dim1=16, input_dim2=16, n_layers=8, conv_units=128, dense_units=256)
model_res.fit(xtrain_deep, ytrain_deep, epochs=2, batch_size=2048)
model_res.evaluate(xtest_deep, ytest_deep, batch_size=10240)
model_res.plot_acc_curve()
class lstmNet(object):
    def __init__(self, n_classes=2, input_dim1=None, input_dim2=None, n_layers=3, use_dropout=True, drop_ratio=.5,
                 use_bidirec=False, use_gru=False, rnn_units=64, use_dense=True, dense_units=64, use_batch=True,
                 metrics='accuracy', optimizer='rmsprop'):
        self.n_classes = n_classes
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_layers = n_layers
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_bidierc = use_bidirec
        self.use_gru = use_gru
        self.rnn_units = rnn_units
        self.use_dense = use_dense
        self.use_batch = use_batch
        self.dense_units = dense_units
        self.metrics = metrics
        self.optimizer = optimizer
        self.model = self._init_model()

    def _init_model(self):
        inputs = Input(shape=(self.input_dim1, self.input_dim2))

        def _lstm_block(layers, name_index=None):
            if self.use_bidierc:
                res = Bidirectional(LSTM(self.rnn_units, return_sequences=True,
                                         recurrent_dropout=self.drop_ratio), name='bidi_lstm_'+str(name_index))(layers)
            elif self.use_gru:
                res = GRU(self.rnn_units, return_sequences=True,
                          recurrent_dropout=self.drop_ratio, name='gru_'+str(name_index))(layers)
            else:
                res = LSTM(self.rnn_units, return_sequences=True,
                           recurrent_dropout=self.drop_ratio, name='lstm_'+str(name_index))(layers)

            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)

            return res

        # No matter for LSTM, GRU, bidirection LSTM, final layer can not use 'return_sequences' output.
        for i in range(self.n_layers - 1):
            if i == 0:
                res = _lstm_block(inputs, name_index=i)
            else:
                res = _lstm_block(res, name_index=i)

        # final LSTM layer
        if self.use_bidierc:
            res = Bidirectional(LSTM(self.rnn_units), name='bire_final')(res)
        elif self.use_gru:
            res = GRU(self.rnn_units, name='gru_final')(res)
        else:
            res = LSTM(self.rnn_units, name='lstm_final')(res)

        # whether or not to use Dense layer
        if self.use_dense:
            res = Dense(self.dense_units, name='dense_1')(res)
            if self.use_batch:
                res = BatchNormalization(name='batch_1')(res)
            res = Activation('relu')(res)
            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid', name='out')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax', name='out')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameter n_class must be provide up or equals to 2!')

        return model

    def fit(self, data, label, epochs=100, batch_size=256):
        #label = check_label_shape(label)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1,
                                  validation_data=(xvalidate, yvalidate))
        print('Model evaluation on validation datasets accuracy:{:.2f}'.format(
            self.model.evaluate(xvalidate, yvalidate)[1]*100))
        return self

    def evaluate(self, data, label, batch_size=None, silent=False):
        #label = check_label_shape(label)

        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.2f}'.format(acc*100))
        return acc

    def plot_acc_curve(self, plot_acc=True, plot_loss=True, figsize=(8, 6)):
        style.use('ggplot')

        if plot_acc:
            fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax1.plot(self.his.history['acc'], label='Train accuracy')
            ax1.plot(self.his.history['val_acc'], label='Validation accuracy')
            ax1.set_title('Train and validation accuracy curve')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy score')
            plt.legend()

        if plot_loss:
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            ax2.plot(self.his.history['loss'], label='Train Loss')
            ax2.plot(self.his.history['val_loss'], label='Validation Loss')
            ax2.set_title('Train and validation loss curve')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss score')
            plt.legend()

        plt.show()

model_lstm = lstmNet(n_classes=2, input_dim1=16, input_dim2=16, n_layers=3)
model_lstm.fit(xtrain_deep, ytrain_deep, epochs=2, batch_size=4086)
model_lstm.evaluate(xtest_deep, ytest_deep, batch_size=4092)
model_lstm.plot_acc_curve()
class denseNet(object):
    def __init__(self, input_dim1=None, input_dim2=None, n_classes=2, basic_residual=False, n_layers=4, flatten=True, use_dense=True,
                 n_dense_layers=1, conv_units=64, stride=1, padding='SAME', dense_units=128, drop_ratio=.5,
                 optimizer='rmsprop', metrics='accuracy'):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_classes = n_classes
        self.basic_residual = basic_residual
        self.n_layers = n_layers
        self.flatten = flatten
        self.use_dense = use_dense
        self.n_dense_layers = n_dense_layers
        self.conv_units = conv_units
        self.stride = stride
        self.padding = padding
        self.dense_units = dense_units
        self.drop_ratio = drop_ratio
        self.optimizer = optimizer
        self.metrics = metrics
        self.model = self._init_model()

    # this will build DenseNet or ResidualNet structure, this model is already compiled.
    def _init_model(self):
        inputs = Input(shape=(self.input_dim1, self.input_dim2))

        # dense net residual block
        def _res_block(layers, added_layers=inputs):
            res = Conv1D(self.conv_units, self.stride, padding=self.padding)(layers)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            res = Conv1D(self.input_dim2, self.stride, padding=self.padding)(res)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            if self.basic_residual:
                return keras.layers.add([res, layers])
            else:
                return keras.layers.add([res, added_layers])

        # construct residual block chain.
        for i in range(self.n_layers):
            if i == 0:
                res = _res_block(inputs)
            else:
                res = _res_block(res)

        # using flatten or global average pooling to process Convolution result
        if self.flatten:
            res = Flatten()(res)
        else:
            res = GlobalAveragePooling1D()(res)

        # whether or not use dense net, also with how many layers to use
        if self.use_dense:
            for j in range(self.n_dense_layers):
                res = Dense(self.dense_units)(res)
                res = BatchNormalization()(res)
                res = Activation('relu')(res)
                res = Dropout(self.drop_ratio)(res)

        if self.n_classes == 2:
            out = Dense(self.n_classes, activation='sigmoid')(res)
            model = Model(inputs, out)
            print('Model structure:')
            model.summary()
            model.compile(loss='binary_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        elif self.n_classes > 2:
            out = Dense(self.n_classes, activation='softmax')(res)
            model = Model(inputs, out)
            print('Model Structure:')
            model.summary()
            model.compile(loss='categorical_crossentropy', metrics=[self.metrics], optimizer=self.optimizer)
        else:
            raise AttributeError('parameters n_classes must up to 2!')

        return model

    # Fit on given training data and label. Here I will auto random split the data to train and validation data,
    # for test datasets, I will just use it if model already trained then I will evaluate the model.
    def fit(self, data, label, epochs=100, batch_size=256):
        # self.model = self._init_model()
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
        self.his = self.model.fit(xtrain, ytrain, verbose=1, epochs=epochs,
                             validation_data=(xvalidate, yvalidate), batch_size=batch_size)
        print('After training, model accuracy on validation datasets is {:.4f}'.format(self.model.evaluate(xvalidate, yvalidate)[1]))
        return self

    # evaluate model on test datasets.
    def evaluate(self, data, label, batch_size=None, silent=False):
        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.6f}'.format(acc))
        return acc

    # plot after training accuracy and loss curve.
    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['acc'], label='Train Accuracy')
        ax.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax.set_title('Train and Validation Accruacy Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['loss'], label='Traing Loss')
        ax.plot(self.his.history['val_loss'], label='Validation Loss')
        ax.set_title('Train and Validation Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss score')

        plt.legend()
        plt.show()

model_dense = denseNet(n_classes=2, input_dim1=16, input_dim2=16, n_layers=3, n_dense_layers=2, optimizer='sgd')
model_dense.fit(xtrain_deep, ytrain_deep, epochs=2, batch_size=2048)
model_dense.evaluate(xtest_deep, ytest_deep, batch_size=4092)
model_dense.plot_acc_curve()
