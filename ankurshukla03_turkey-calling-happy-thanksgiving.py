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
import pandas as pd
import os
from keras.models import Model, Sequential
from keras.layers import Dense, Bidirectional,CuDNNLSTM, LSTM, BatchNormalization, Dropout, Input, Conv1D, Activation,CuDNNGRU, Reshape, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train.head()
test.head()
sample_submission.head()
print(train.shape)
print(test.shape)
turkey = len(train[train['is_turkey']==1])
not_turkey = len(train[train['is_turkey']==0])
print("Number of turkey sound :",turkey)
print("Number of not turkey sound :",not_turkey)
print("percentage of turkey sound {0:.2f}".format(turkey/train.shape[0]))
print("percentage of not turkey sound {0:.2f}".format(not_turkey/train.shape[0]))
from IPython.display import YouTubeVideo
row = 3
YouTubeVideo(train['vid_id'][row],start=train['start_time_seconds_youtube_clip'][row],end=train['end_time_seconds_youtube_clip'][row])
print(train['audio_embedding'].head())

#see the possible list lengths of the first dimension
print("train's audio_embedding can have this many frames: "+ str(train['audio_embedding'].apply(lambda x: len(x)).unique())) 
print("test's audio_embedding can have this many frames: "+ str(test['audio_embedding'].apply(lambda x: len(x)).unique())) 

#see the possible list lengths of the first element
print("each frame can have this many features: "+str(train['audio_embedding'].apply(lambda x: len(x[0])).unique()))
train_train, train_val = train_test_split(train)
xtrain = train_train['audio_embedding'].tolist()
ytrain = train_train['is_turkey'].values

xval = train_val['audio_embedding'].tolist()
yval = train_val['is_turkey'].values

x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)
def first_model():
    inp = Input((10, 128))
    x = Conv1D(512, 10, padding='same')(inp)
    x = Conv1D(256, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(512, return_sequences=True, recurrent_dropout=0.1))(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, 10, padding='same')(x)
    x = Conv1D(128, 5, padding='same')(x)
    x = Bidirectional(LSTM(512, return_sequences=True, recurrent_dropout=0.1))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    print(model.summary())
    return model
def model2():
    inp = Input(shape=(10, 128))
    x = Conv1D(128, 1, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Bidirectional(CuDNNGRU(256, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    concat = concatenate([avg_pool, max_pool])
    concat = Dense(64, activation="relu")(concat)
    concat = Dropout(0.5)(concat)
    output = Dense(1, activation="sigmoid")(concat)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    print(model.summary())
    return model
# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
def model3():
    model = Sequential()
    model.add(BatchNormalization(momentum=0.90,input_shape=(10, 128)))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
    model.add(Bidirectional(CuDNNLSTM(1, return_sequences = True)))
    model.add(Attention(10))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
batch_size = 100
epochs = 200
model = first_model()
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)

early_stopping = EarlyStopping(monitor='val_loss',
                              patience=8,
                              verbose=1,
                              mode='min',
                              restore_best_weights=True)

callback = [learning_rate_reduction,early_stopping]
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,validation_data=(x_val, y_val), callbacks=callback, verbose=2)

score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)
print('Test accuracy:', acc)
#plt.figure(figsize=(12,8))
#plt.plot(range(1, epochs+1), history.history['loss'], label='Train Accuracy')
#plt.plot(range(1, epochs+1), history.history['val_loss'], label='Validation Accuracy')
#plt.legend()
#plt.show()
test_data = test['audio_embedding'].tolist()
submission = model.predict(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
submission['is_turkey'] = submission.is_turkey.round(0).astype(int)
print(submission.head(20))
submission.to_csv('submission6.csv', index=False)
