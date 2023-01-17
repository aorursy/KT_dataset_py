import numpy as np
import pandas as pd
import os
from keras.models import Model, load_model
from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout, Input, Conv1D, Add, Conv2D, Activation, Flatten, Reshape, MaxPooling2D, AveragePooling2D
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
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
    
def pad_with_same(data, maxlen=10):
    res = []
    for x in data:
        l = len(x)
        if l == maxlen:
            res.append(x)
        else:
            while l < maxlen:
                x = np.concatenate([x[:maxlen-l],x],axis=0)
                l = len(x)
            res.append(x)
    return np.array(res).astype(np.float32) / 255
# train_train, train_val = train_test_split(train)
xtrain = train['audio_embedding'].tolist()
ytrain = train['is_turkey'].values

# xval = train_val['audio_embedding'].tolist()
# yval = train_val['is_turkey'].values

x_train = pad_with_same(xtrain, maxlen=10)
# x_val = pad_with_same(xval, maxlen=10)

y_train = np.asarray(ytrain)
# y_val = np.asarray(yval)

batch_size = 256
epochs = 50

inp = Input((10, 128))
x = Conv1D(128, 1, padding='same')(inp)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(x)
# x = Conv1D(64, 1, padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Bidirectional(LSTM(64, activation='relu'))(x)
# x = Attention(10)(x)
x = Conv1D(20, 1, padding='same')(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x1 = Dense(1, activation='sigmoid')(x)

x = Conv1D(256, 1, padding='same')(inp)
x = Reshape((16,16,10))(x)
x = Conv2D(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(128, 3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
# x = Conv2D(256, 3, padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x2 = Dense(1, activation='sigmoid')(x)

x = Add()([x1, x2])
x2 = Dense(1, activation='sigmoid')(x)

model = Model(inp, x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)
checkpoint = ModelCheckpoint('out.h5', monitor='val_acc', verbose=0, save_best_only=True)

model.fit(x_train, y_train,
          validation_split=0.2,
          batch_size=batch_size,
          nb_epoch=epochs, callbacks=[checkpoint])

score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)
print('Test accuracy:', acc)
model = load_model('out.h5')

test_data = test['audio_embedding'].tolist()
submission = model.predict(pad_with_same(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
submission['is_turkey'] = submission.is_turkey.round(0).astype(int)
print(submission.head(40))
submission.to_csv('submission.csv', index=False)

