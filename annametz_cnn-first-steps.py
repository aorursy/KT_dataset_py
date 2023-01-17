!pip install hyperas
import pandas as pd

import numpy as np

np.random.seed(0)



from sklearn.model_selection import train_test_split



from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim

from hyperas.distributions import choice, uniform



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.utils import to_categorical

from keras import optimizers



import matplotlib.pyplot as plt

%matplotlib inline
def data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

#     print(train.shape, test.shape)

    # review

    Y = train['label']

    X = train.drop(axis=1,columns='label')

    X = X.values.reshape(-1,28,28,1) # convert to ndarray

    Y = Y.values

    test = test.values.reshape(-1,28,28,1)

#     print(X.shape)

    # normalization

    if np.max(X) > 1:

        X = X / 255

    if np.max(test) > 1:

        test = test / 255

#     print(np.max(X), np.max(test))

    # categorization

    if len(Y.shape) == 1:

        num_classes = len(set(Y))

        Y = to_categorical(Y, num_classes)

#     print(num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

    return X_train, X_test, y_train, y_test



# data()
num_classes = 10
model_1 = Sequential([

    Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)),

    MaxPool2D(pool_size=(2, 2)),

    

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

    MaxPool2D(pool_size=(2, 2)),

    

    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

    Dropout(0.25),

    

    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),

    Dropout(0.25),

    

    Flatten(),

    Dense(512, activation='relu'),

    

    Dense(num_classes, activation='softmax')

])



model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_1.summary()
model_2 = Sequential([

    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    MaxPool2D(pool_size=(2, 2), strides=2),

    

    Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),

    MaxPool2D(pool_size=(2, 2),  strides=2),

    

    Flatten(),

    

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(num_classes, activation='softmax')

])



model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_2.summary()
model_3 = Sequential([

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    MaxPool2D(pool_size=(2, 2), strides=2),

    

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

    MaxPool2D(pool_size=(2, 2),  strides=2),

    

    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

    MaxPool2D(pool_size=(2, 2),  strides=2),

    

    Flatten(),

    

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(num_classes, activation='softmax')

])



optimizer = optimizers.Adam(lr=0.0001)

model_3.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model_3.summary()
model_4 = Sequential([

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),

    MaxPool2D(pool_size=(2, 2), strides=2),

    Dropout(0.25),

    

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

    MaxPool2D(pool_size=(2, 2),  strides=2),

    Dropout(0.25),

    

    Flatten(),

    

    Dense(256, activation='relu'),

    Dense(128, activation='relu'),

    Dense(num_classes, activation='softmax')

])



optimizer = optimizers.Adam(lr=0.0001)

model_4.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model_4.summary()
X_train, X_test, y_train, y_test = data()
model = model_4

model.fit(

    X_train, y_train,

    batch_size=128,

    epochs=10,    

)
score = model.evaluate(X_test, y_test, verbose=0)

score
predictions = model.predict(test)

predictions = np.argmax(predictions, axis=1)
def model(X_train, Y_train, X_val, Y_val):

    num_classes = 10

    

    model = Sequential()

    model_choice = {{choice(['model_3', 'model_4'])}}

    if model_choice == 'model_3':

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))

        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPool2D(pool_size=(2, 2),  strides=2))

        model.add(Dropout({{uniform(0, 1)}}))

        

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPool2D(pool_size=(2, 2),  strides=2))

        model.add(Dropout({{uniform(0, 1)}}))

        

        model.add(Flatten())

        

        model.add(Dense(128), activation='relu')

        model.add(Dense({{choice([32, 64])}}, activation='relu'))

        

    

    elif model_choice == 'model_4':

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))

        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Dropout({{uniform(0, 1)}}))

    

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        model.add(MaxPool2D(pool_size=(2, 2),  strides=2))

        model.add(Dropout({{uniform(0, 1)}}))

    

        model.add(Flatten())

        

        model.add(Dense({{choice([128, 256])}}, activation='relu'))

        model.add(Dense({{choice([64, 128])}}, activation='relu'))

        

    model.add(Dense(num_classes, activation='softmax'))

    

    adam = keras.optimizers.Adam(lr=0.0001)

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    

    model.fit(X_train, y_train,

              batch_size=128,

              nb_epoch=10,

              verbose=2,

              validation_data=(X_test, y_test))

    

    score, acc = model.evaluate(X_test, y_test, verbose=0)

    print('Val accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
best_run, best_model = optim.minimize(model=model,

                                      data=data,

                                      algo=tpe.suggest,

                                      max_evals=30,

                                      trials=Trials(),

                                      notebook_name='CNN_first_steps')
submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':predictions })



submission.to_csv("cnn_architecture.csv",index=False)