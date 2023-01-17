import numpy as np

from numpy import genfromtxt



d = genfromtxt('../input/train.csv', delimiter=',', dtype='float32',skip_header=1)



num_labels = 10

labels = np.ndarray(shape=(len(d), num_labels))

data = np.ndarray(shape=(len(d), len(d[0][1:])))

idx = 0

for l in d:

    labels[idx] = (np.arange(num_labels) == l[0]).astype(np.float32)

    data[idx] = l[1:]

    idx = idx + 1    
split = int(len(data)*.3)

val_data = data[:split]/255.

val_labels = labels[:split]

train_data = data[split:]/255

train_labels = labels[split:]



train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')

val_data = val_data.reshape(val_data.shape[0], 28, 28, 1).astype('float32')
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten

from keras.optimizers import SGD



def model1():

    return Sequential([

        Dense(32, input_dim=28*28),

        Activation('relu'),

        Dense(10),

        Activation('softmax'),

    ])



def model2(): # 99/98

    model = Sequential()

    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(28, 28, 1), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_labels, activation='softmax'))

    return model



def model3(): 

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(28,28,1)))

    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))

    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    return model



model = model1()

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



model.fit(data[split:]/255, train_labels,

          nb_epoch=20,

          batch_size=16,

          verbose=2)
train_pred = model.predict ( data[split:]/255 )

val_pred = model.predict(data[:split]/255)
np.sum(np.all(np.round(train_pred) == train_labels, axis=1))/len(train_pred)
model.evaluate(train_data, train_labels, batch_size=16, verbose=0)
np.sum(np.all(np.round(val_pred) == val_labels, axis=1))/len(val_pred)
model.evaluate(val_data, val_labels, batch_size=16, verbose=0)