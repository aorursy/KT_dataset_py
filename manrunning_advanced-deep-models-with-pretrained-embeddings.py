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
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, LSTM, GRU, BatchNormalization
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, Flatten, Bidirectional
from tensorflow.keras.layers import Embedding, Reshape, CuDNNGRU, CuDNNLSTM
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from scikitplot.metrics import plot_confusion_matrix

# load train and test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train datasets shape:", train.shape)
print("Test datasets shape:", test.shape)
# Show some train datasets 
print('Train data samples:')
train.head()
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
max_length = 100
max_features = 50000
token = Tokenizer(num_words=max_features)
token.fit_on_texts(list(np.asarray(train_data.question_text)))
xtrain = token.texts_to_sequences(np.asarray(train_data.question_text))
xvalidate = token.texts_to_sequences(np.asarray(validation_data.question_text))
xtest = token.texts_to_sequences(np.asarray(test_new.question_text))

# Because Tokenizer will split the sentence, for some sentence are smaller,
# so we have to pad the missing position
xtrain = pad_sequences(xtrain, maxlen=max_length)
xvalidate = pad_sequences(xvalidate, maxlen=max_length)
xtest = pad_sequences(xtest, maxlen=max_length)

ytrain = train_data.target
yvaliate = validation_data.target
# Here I write a helper function to evaluate model
def evaluate(y, pred):
    f1_list = list()
    thre_list = np.arange(0.1, 0.501, 0.01)
    for thresh in thre_list:
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(y, (pred>thresh).astype(int))
        f1_list.append(f1)
        print("F1 score at threshold {0} is {1}".format(thresh, f1))
    #return f1_list
    plot_confusion_matrix(y, np.array(pd.Series(pred.reshape(-1,)).map(lambda x:1 if x>thre_list[np.argmax(f1_list)] else 0)))
    print('Best Threshold: ',thre_list[np.argmax(f1_list)])
    return thre_list[np.argmax(f1_list)]
# Here I will build a DNN model as deep learning baseline

# Here I write a DNN class for many other cases, 
# you can choose how many layers, how many units, whether to use dropout,
# whether to use batchnormalization, also with optimizer! 
class dnnNet(object):
    def __init__(self, n_classes=2, n_dims=None, n_layers=3, n_units=64, use_dropout=True, drop_ratio=.5, use_batchnorm=True,
                 metrics='accuracy', optimizer='rmsprop', use_em=False, em_weights=None, em_input_dim=None, fit_split=False):
        self.n_classes = n_classes
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_units = n_units
        self.use_dropout = use_dropout
        self.drop_ratio = drop_ratio
        self.use_batchnorm = use_batchnorm
        self.metrics = metrics
        self.optimizer = optimizer
        self.use_em = use_em
        self.em_weights = em_weights
        self.em_input_dim = em_input_dim
        self.fit_split = fit_split
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
            out = Dense(1, activation='sigmoid')(res)
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
    def fit(self, data, label, epochs=100, batch_size=256, vali_data=None, vali_label=None):
        if self.fit_split:
            xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
            self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1,
                                      validation_data=(xvalidate, yvalidate))
            print('Model evaluation on validation datasets accuracy:{:.4f}'.format(self.model.evaluate(xvalidate, yvalidate)[1]))
        else: 
            self.his = self.model.fit(data, label, epochs=epochs, batch_size=batch_size, verbose=1,
                                      validation_data=(vali_data, vali_label))
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
class residualNet(object):
    def __init__(self, input_dim1=None, input_dim2=None, n_classes=2, n_layers=4, flatten=True, use_dense=True,
                 n_dense_layers=1, conv_units=64, stride=1, padding='SAME', dense_units=128, drop_ratio=.5,
                 optimizer='rmsprop', metrics='accuracy', use_em=False, em_weights=None, fit_split=False):
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
        self.use_em = use_em
        self.em_weights = em_weights
        self.fit_split = fit_split
        self.model = self._init_model()

    def _init_model(self):
        
        if self.use_em:
            inputs = Input(shape=(self.em_weights.shape[1], ))
        else: inputs = Input(shape=(self.input_dim1, self.input_dim2))

        # dense net residual block
        def _res_block(layers):
            res = Conv1D(self.conv_units, self.stride, padding=self.padding)(layers)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            if self.use_em:
                res = Conv1D(self.em_weights.shape[1], self.stride, padding=self.padding)(res)
            else: res = Conv1D(self.input_dim2, self.stride, padding=self.padding)(res)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)

            return keras.layers.add([layers, res])

        # construct residual block chain.
        for i in range(self.n_layers):
            if i == 0:
                if self.use_em:
                    res = Embedding(max_features, em_size)(inputs)
                    res = _res_block(res)
                else: res = _res_block(inputs)
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
            out = Dense(1, activation='sigmoid')(res)
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

        if self.fit_split:
            xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
            self.his = self.model.fit(xtrain, ytrain, verbose=1, epochs=epochs,
                                      validation_data=(xvalidate, yvalidate), batch_size=batch_size)
            print('After training, model accuracy on validation datasets is {:.2f}%'.format(
                self.model.evaluate(xvalidate, yvalidate)[1]*100))
        else: self.his = self.model.fit(data, label, batch_size=batch_size, epochs=epochs, verbose=1)
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
        if self.fit_split:
            ax.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax.set_title('Train and Validation Accruacy Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['loss'], label='Traing Loss')
        if self.fit_split:
            ax.plot(self.his.history['val_loss'], label='Validation Loss')
        ax.set_title('Train and Validation Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss score')

        plt.legend()
        plt.show()

class lstmNet(object):
    def __init__(self, n_classes=2, input_dim1=None, input_dim2=None, n_layers=3, use_dropout=True, drop_ratio=.5,
                 use_bidirec=False, use_gru=False, rnn_units=64, use_dense=True, dense_units=64, use_batch=True,
                 metrics='accuracy', optimizer='rmsprop', use_em=False, em_weights=None, em_input_dim=None, fit_split=False):
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
        self.use_em = use_em
        self.em_weights = em_weights
        self.em_input_dim = em_input_dim
        self.fit_split = fit_split
        self.is_gpu = tf.test.is_gpu_available()
        self.model = self._init_model()

    def _init_model(self):
        
        if self.use_em:
            inputs =  Input(shape=(self.em_input_dim, ))
        else: inputs = Input(shape=(self.input_dim1, self.input_dim2))

        def _lstm_block(layers, name_index=None):
            if self.use_bidierc:
                if self.is_gpu:
                    res = Bidirectional(CuDNNLSTM(self.rnn_units, return_sequences=True),
                                        name='bidi_lstm_'+str(name_index))(layers)
                else: res = Bidirectional(LSTM(self.rnn_units, return_sequences=True,
                                         recurrent_dropout=self.drop_ratio), name='bidi_lstm_'+str(name_index))(layers)
            elif self.use_gru:
                if self.is_gpu:
                    res = CuDNNGRU(self.rnn_units, return_sequences=True)(layers)
                else: res = GRU(self.rnn_units, return_sequences=True,
                          recurrent_dropout=self.drop_ratio, name='gru_'+str(name_index))(layers)
            else:
                if self.is_gpu:
                    res = CuDNNLSTM(self.rnn_units, return_sequences=True)(layers)
                else: res = LSTM(self.rnn_units, return_sequences=True,
                           recurrent_dropout=self.drop_ratio, name='lstm_'+str(name_index))(layers)

            if self.use_dropout:
                res = Dropout(self.drop_ratio)(res)

            return res

        # No matter for LSTM, GRU, bidirection LSTM, final layer can not use 'return_sequences' output.
        for i in range(self.n_layers - 1):
            if i == 0:
                if self.use_em:
                    res = Embedding(max_features, em_size)(inputs)
                    res = _lstm_block(res, name_index=i)
                else: res = _lstm_block(inputs, name_index=i)
            else:
                res = _lstm_block(res, name_index=i)

        # final LSTM layer
        if self.use_bidierc:
            if self.is_gpu:
                res = Bidirectional(CuDNNLSTM(self.rnn_units))(res)
            else: res = Bidirectional(LSTM(self.rnn_units), name='bire_final')(res)
        elif self.use_gru:
            if self.is_gpu:
                res = CuDNNGRU(self.rnn_units)(res)
            else:res = GRU(self.rnn_units, name='gru_final')(res)
        else:
            if self.is_gpu:
                res = CuDNNLSTM(self.rnn_units)(res)
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
            out = Dense(1, activation='sigmoid', name='out')(res)
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

    def fit(self, data, label, epochs=100, batch_size=256, vali_data=None, vali_label=None):
        #label = check_label_shape(label)
        if self.fit_split:
            xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
            self.his = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1,
                                      validation_data=(xvalidate, yvalidate))
            print('Model evaluation on validation datasets accuracy:{:.2f}'.format(
                self.model.evaluate(xvalidate, yvalidate)[1]*100))
        else: self.his = self.model.fit(data, label,
                                        epochs=epochs, batch_size=batch_size, verbose=1, 
                                        validation_data=(vali_data,vali_label))
        return self

    def evaluate(self, data, label, batch_size=None, silent=False):
        #label = check_label_shape(label)

        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.2f}'.format(acc*100))
        return acc
    
    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)    

    def plot_acc_curve(self, plot_acc=True, plot_loss=True, figsize=(8, 6)):
        style.use('ggplot')

        if plot_acc:
            fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax1.plot(self.his.history['acc'], label='Train accuracy')
            if self.fit_split:
                ax1.plot(self.his.history['val_acc'], label='Validation accuracy')
            ax1.set_title('Train and validation accuracy curve')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy score')
            plt.legend()

        if plot_loss:
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            ax2.plot(self.his.history['loss'], label='Train Loss')
            if self.fit_split:
                ax2.plot(self.his.history['val_loss'], label='Validation Loss')
            ax2.set_title('Train and validation loss curve')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss score')
            plt.legend()

        plt.show()

class denseNet(object):
    def __init__(self, input_dim1=None, input_dim2=None, n_classes=2, basic_residual=False, n_layers=4, flatten=True, use_dense=True,
                 n_dense_layers=1, conv_units=64, stride=1, padding='SAME', dense_units=128, drop_ratio=.5,
                 optimizer='rmsprop', metrics='accuracy', use_em=False, em_weights=None, em_input_dim=None, fit_split=False):
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
        self.use_em = use_em
        self.em_weights = em_weights
        self.em_input_dim = em_input_dim
        self.fit_split = fit_split
        self.model = self._init_model()

    # this will build DenseNet or ResidualNet structure, this model is already compiled.
    def _init_model(self):
        
        if self.use_em:
            inputs = Input(shape=(self.em_input_dim, ))
        else:
            inputs = Input(shape=(self.input_dim1, self.input_dim2))

        # dense net residual block
        def _res_block(layers, added_layers=inputs):
            res = Conv1D(self.conv_units, self.stride, padding=self.padding)(layers)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)
            res = Dropout(self.drop_ratio)(res)
            
            if self.use_em:
                res = Conv1D(self.em_input_dim, self.stride, padding=self.padding)(res)
            else: res = Conv1D(self.input_dim2, self.stride, padding=self.padding)(res)
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
                if self.use_em:
                    res = Embedding(max_features, em_size, weights=[self.em_weights])(inputs)
                    res = _res_block(res)
                else: res = _res_block(inputs)
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
            out = Dense(1, activation='sigmoid')(res)
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
    def fit(self, data, label, epochs=100, batch_size=256, vali_data=None, vali_label=None):
        # self.model = self._init_model()
        if self.fit_split:   # Whether or not to split the training data
            xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)
            self.his = self.model.fit(xtrain, ytrain, verbose=1, epochs=epochs,
                                 validation_data=(xvalidate, yvalidate), batch_size=batch_size)
            print('After training, model accuracy on validation datasets is {:.4f}'.format(self.model.evaluate(xvalidate, yvalidate)[1]))
        else:    
            self.his = self.model.fit(data, label, verbose=1, 
                                      epochs=epochs, batch_size=batch_size, validation_data=(vali_data, vali_label))
        return self

    # evaluate model on test datasets.
    def evaluate(self, data, label, batch_size=None, silent=False):
        acc = self.model.evaluate(data, label, batch_size=batch_size)[1]
        if not silent:
            print('Model accuracy on Testsets : {:.6f}'.format(acc))
        return acc
    
    def predict(self, data, batch_size=None):
        return self.model.predict(data, batch_size=batch_size)
    
    # plot after training accuracy and loss curve.
    def plot_acc_curve(self):
        style.use('ggplot')

        fig1, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['acc'], label='Train Accuracy')
        if self.fit_split:
            ax.plot(self.his.history['val_acc'], label='Validation Accuracy')
        ax.set_title('Train and Validation Accruacy Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy score')
        plt.legend()

        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(self.his.history['loss'], label='Traing Loss')
        if self.fit_split:
            ax.plot(self.his.history['val_loss'], label='Validation Loss')
        ax.set_title('Train and Validation Loss Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss score')

        plt.legend()
        plt.show()

del train_new, train, train_data, validation_data
import gc
gc.collect()
time.sleep(10)
em_file = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*d.split(' ')) for d in open(em_file))

all_embs = np.stack(embedding_index.values())
em_mean, em_std = all_embs.mean(), all_embs.std()
em_size = all_embs.shape[1]

word_index = token.word_index
nb_words = min(max_features, len(word_index))
em_matrix = np.random.normal(em_mean, em_std, (nb_words, em_size))
# loop for every word
for word, i in word_index.items():
    if i >= max_features: continue
    em_v = embedding_index.get(word)
    if em_v is not None:
        em_matrix[i] = em_v
    

# # Before fitting model, convert label to be 2D
# ytrain_deep = keras.utils.to_categorical(ytrain)
# yvaliate_deep = keras.utils.to_categorical(yvaliate)
# # Build DNN model
# model_dnn = dnnNet(n_classes=2, n_dims=max_length, use_em=True, n_layers=3, n_units=512,
#                    em_weights=em_matrix, em_input_dim=em_matrix.shape[1])
# # Fit DNN model
# model_dnn.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

# # Evaluate DNN model based on validataion data
# model_dnn.evaluate(xvalidate, yvaliate)

# # Plot learning curve and validate curve
# model_dnn.plot_acc_curve()

# # Use trained model to make prediction based on validation data
# pred_dnn = model_dnn.predict(xvalidate, batch_size=4092)

# # Evaluate model result based on different threshold
# evaluate(yvaliate, pred_dnn)
# model_dense = denseNet(n_classes=2, use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')

# model_dense.fit(xtrain, ytrain , epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

# model_dense.evaluate(xvalidate, yvaliate, batch_size=10240)

# pred_dense = model_dense.predict(xvalidate)

# evaluate(yvaliate, pred_dense)
model_lstm = lstmNet(n_classes=2, drop_ratio=.1,
                     use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')
model_lstm.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

pred_lstm_glove = model_lstm.predict(xvalidate, batch_size=10240)
evaluate(yvaliate, pred_lstm_glove)

model_bidi_lstm = lstmNet(n_classes=2, use_bidirec=True, dense_units=128, drop_ratio=.1,
                          use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')

model_bidi_lstm.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

pred_bidi_lstm_glove = model_lstm.predict(xvalidate, batch_size=10240)
evaluate(yvaliate, pred_bidi_lstm_glove)

lstm_glove = model_lstm.predict(xtest)
bidi_lstm_glove = model_bidi_lstm.predict(xtest)
del model_lstm, model_bidi_lstm
del embedding_index, all_embs, word_index, em_matrix
gc.collect()
time.sleep(10)
em_file = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(em_file) if len(o)>100)

all_embs = np.stack(embedding_index.values())
em_mean, em_std = all_embs.mean(), all_embs.std()
em_size = all_embs.shape[1]

word_index = token.word_index
nb_words = min(max_features, len(word_index))
em_matrix = np.random.normal(em_mean, em_std, (nb_words, em_size))
# loop for every word
for word, i in word_index.items():
    if i >= max_features: continue
    em_v = embedding_index.get(word)
    if em_v is not None:
        em_matrix[i] = em_v
    

model_lstm = lstmNet(n_classes=2, drop_ratio=.1,
                     use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')
model_lstm.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

pred_lstm_wiki = model_lstm.predict(xvalidate, batch_size=10240)
evaluate(yvaliate, pred_lstm_wiki)

model_bidi_lstm = lstmNet(n_classes=2, use_bidirec=True, dense_units=128, drop_ratio=.1,
                          use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')

model_bidi_lstm.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

pred_bidi_lstm_wiki = model_lstm.predict(xvalidate, batch_size=10240)
evaluate(yvaliate, pred_bidi_lstm_wiki)

lstm_wiki = model_lstm.predict(xtest)
bidi_lstm_wiki = model_bidi_lstm.predict(xtest)
del model_lstm, model_bidi_lstm
del embedding_index, all_embs, word_index, em_matrix
gc.collect()
time.sleep(10)
em_file = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(em_file, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embedding_index.values())
em_mean, em_std = all_embs.mean(), all_embs.std()
em_size = all_embs.shape[1]

word_index = token.word_index
nb_words = min(max_features, len(word_index))
em_matrix = np.random.normal(em_mean, em_std, (nb_words, em_size))
# loop for every word
for word, i in word_index.items():
    if i >= max_features: continue
    em_v = embedding_index.get(word)
    if em_v is not None:
        em_matrix[i] = em_v
    

model_lstm = lstmNet(n_classes=2, drop_ratio=.1,
                     use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')
model_lstm.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

pred_lstm_para = model_lstm.predict(xvalidate, batch_size=10240)
evaluate(yvaliate, pred_lstm_para)

model_bidi_lstm = lstmNet(n_classes=2, use_bidirec=True, dense_units=128, drop_ratio=.1,
                          use_em=True, em_weights=em_matrix, em_input_dim=max_length, optimizer='adam')

model_bidi_lstm.fit(xtrain, ytrain, epochs=2, batch_size=512, vali_data=xvalidate, vali_label=yvaliate)

pred_bidi_lstm_para = model_lstm.predict(xvalidate, batch_size=10240)
evaluate(yvaliate, pred_bidi_lstm_para)

lstm_para = model_lstm.predict(xtest)
bidi_lstm_para = model_bidi_lstm.predict(xtest)
del model_lstm, model_bidi_lstm
del embedding_index, all_embs, word_index, em_matrix
gc.collect()
time.sleep(10)
model_num = 6
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
pred_result = np.empty([len(pred_lstm_glove), model_num])
pred_result[:, 0] = pred_lstm_glove.reshape(-1, )
pred_result[:, 1] = pred_bidi_lstm_glove.reshape(-1, )
pred_result[:, 2] = pred_lstm_wiki.reshape(-1, )
pred_result[:, 3] = pred_bidi_lstm_wiki.reshape(-1, )
pred_result[:, 4] = pred_lstm_para.reshape(-1, )
pred_result[:, 5] = pred_bidi_lstm_para.reshape(-1, )

lr.fit(pred_result, yvaliate)

weights = lr.coef_

# Here is just used for get combining result, and best threshold.
sub_pred_weighted = np.sum([pred_result[:, i]*weights[i] for i in range(model_num)], axis=0)
best_thre = evaluate(yvaliate ,sub_pred_weighted)
# GET different model prediction result on test datasets
sub_data = np.empty([len(xtest), model_num])
sub_data[:, 0] = lstm_glove.reshape(-1, )
sub_data[:, 1] = bidi_lstm_glove.reshape(-1, )
sub_data[:, 2] = lstm_wiki.reshape(-1, )
sub_data[:, 3] = bidi_lstm_glove.reshape(-1, )
sub_data[:, 4] = lstm_para.reshape(-1, )
sub_data[:, 5] = bidi_lstm_para.reshape(-1, )

#sub_pred = 0.1 * pred_lstm_glove + 0.2*pred_bidi_lstm_glove + 0.1*pred_lstm_wiki + 0.2*pred_bidi_lstm_wiki+0.1*pred_lstm_para+0.3*pred_lstm_para
# According to Linear Regression model result with different weights multiply with prediction.
sub_pred = np.sum([sub_data[:, i]*weights[i] for i in range(model_num)], axis=0)
sub_pred = (sub_pred > best_thre).astype(int)

sub_df = pd.DataFrame({'qid':test.qid.values})
sub_df['prediction'] = sub_pred
sub_df.to_csv('submission.csv', index=False)

