!ls ../input/Kannada-MNIST

# %tensorflow_version 1.x

import tensorflow

print(tensorflow.__version__)

import tensorflow.keras as keras

print(keras.__version__)
import pandas as pd

train_set = pd.read_csv("../input/Kannada-MNIST/train.csv")

print(type(train_set))

train_array = train_set.values

y_train, x_train = train_array[:, :1], train_array[:, 1:]

# print(y_train, y_train.shape,"\n", x_train, x_train.shape)

print(x_train.dtype, x_train.min(), x_train.max())

test_set = pd.read_csv("../input/Kannada-MNIST/test.csv")

print(type(test_set))

test_array = test_set.values



y_test, x_test = test_array[:, :1], test_array[:, 1:]
print(x_test)

print(x_test.max(), x_test.min())

print(y_test)

print(y_train)
from skimage.io import imshow, imshow_collection



imshow_collection(x_train[:12].reshape(12, 28, 28))

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler((-1, 1), False)

print(scaler.fit(x_train))

# print(scaler.data_max_)

x_train = scaler.transform(x_train)

# print(scaler.data_max_)

# print(x_train)

print(x_train.min(), x_train.max())

print(x_test.min(), x_test.max())

scaler = MinMaxScaler((-1, 1), False)

scaler.fit(x_test)

x_test = scaler.transform(x_test)

print(x_test.min(), x_test.max())
border = 55000

x_train, x_valid, y_train, y_valid = x_train[:border], x_train[border:], y_train[:border], y_train[border:]

from keras.layers import *

from keras.models import Model, Sequential

from keras.callbacks import CSVLogger, ModelCheckpoint



x = Input(shape=(784,))

y = Dense(30, activation=None)(x)

y = Activation('elu')(y)

y = Dropout(rate=0.15)(y)

y = Dense(30, activation=None)(y)

y = Activation('elu')(y)

prediction = Dense(10, activation='softmax')(y)



model = Model(inputs=[x], output=[prediction])



model.compile(optimizer ='sgd',

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
model.fit(

    # x_train - 0.5, y_train,

          x_train, y_train,

          batch_size=16,

          epochs=25,

          verbose=1,

          # validation_data=(x_valid - 0.5, y_valid),

          validation_data=(x_valid, y_valid),

          callbacks=[

              CSVLogger('log.csv'),

              ModelCheckpoint('model.h5', save_best_only=True),

          ])
!ls

print(x_test)
model = keras.models.load_model('model.h5')



# pred_probas = model.predict(x_test - 0.5, batch_size=16)

pred_probas = model.predict(x_test, batch_size=16)



prediction = pred_probas.argmax(axis=1)



# predictions = model.predict(x_test)

# print(predictions)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission['label'] = prediction
submission.head()
submission.to_csv("submission.csv",index=False)
!ls