import numpy as np

import pandas as pd

import os

print(os.listdir('../input'))

import pickle

import matplotlib.pyplot as plt
def unpickle(file):

    with open(file, 'rb') as fo:

        dat = pickle.load(fo, encoding='bytes')

    return dat





def convert_data(batch):

    data = batch[b'data']

    data = data.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])

    X = data/255

    

    labels = batch[b'labels']

    Y = np.eye(10)[labels].astype(int)

    

    return X, Y





def load_data(url):

    file_list = os.listdir(url)

    print(file_list)

    train_set_file = set(file_list) - {'test_batch'}

    test_set_file = 'test_batch'

    

    X_train = np.array([]).reshape(-1, 32, 32, 3)

    Y_train = np.array([]).reshape(-1, 10)

    

    for filename in train_set_file:

        batch = unpickle(url+'/'+filename)

        X, Y = convert_data(batch)

        X_train = np.concatenate((X_train, X))

        Y_train = np.concatenate((Y_train, Y))

    

    batch = unpickle(url+'/'+test_set_file)

    X_test, Y_test = convert_data(batch)

    

    return X_train, Y_train, X_test, Y_test
X_train, Y_train, X_test, Y_test = load_data('../input')



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
Y_categ = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



fig, ax = plt.subplots(4, 8, figsize = (16, 8))

ax = ax.ravel()

for i in range(32):

    ax[i].imshow(X_train[i])

    ax[i].set_title(Y_categ[np.argmax(Y_train[i], axis = 0)])

    ax[i].axis('off')

plt.show()
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, LeakyReLU, BatchNormalization

from keras.optimizers import Adam

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D

from keras.initializers import glorot_normal
# model parameters

input_dim = (32, 32, 3)

output_dim = 10

lr = 0.001

epoch = 15

batch_size = 64





model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                 kernel_initializer=glorot_normal(), input_shape = input_dim))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization())

model.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                 kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                 kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization())

model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                 kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                 kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))

model.add(BatchNormalization())

model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', 

                 kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Flatten())



model.add(Dense(units=512, kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))

model.add(Dropout(rate=0.25))

model.add(BatchNormalization())



model.add(Dense(units=128, kernel_initializer=glorot_normal()))

model.add(LeakyReLU(alpha=0.01))

model.add(Dropout(rate=0.25))

model.add(BatchNormalization())



model.add(Dense(units=output_dim, activation='softmax'))

model.summary()
adam = Adam(lr=lr)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])



hist = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1)
fig, ax = plt.subplots(2, 1)



ax[0].plot(hist.history['loss'])

ax[0].plot(hist.history['val_loss'])

ax[0].legend(['Training', 'Validation'])

ax[1].plot(hist.history['acc'])

ax[1].plot(hist.history['val_acc'])

ax[1].legend(['Training', 'Validation'])

plt.show()
loss, acc = model.evaluate(X_test, Y_test)

print('Test Loss     : {:.8f}'.format(loss))

print('Test Accuracy : {:.4f}'.format(acc))
pred = model.predict_classes(X_test)

actual = np.argmax(Y_test, axis=1)

crosstab = pd.crosstab(pred, actual)

crosstab.columns = pd.Series(Y_categ, name = 'Actual')

crosstab.index = pd.Series(Y_categ, name = 'Pred')

crosstab