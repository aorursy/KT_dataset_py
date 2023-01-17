import numpy

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as k

k.set_image_dim_ordering('th')

from sklearn.utils import shuffle

import pandas as pd

import h5py



#As is good practice, we next initialize the random number seed with a constant 

#to ensure the results are reproducible

seed = 7

numpy.random.seed(seed)
df = pd.read_csv("../input/train.csv")

df = shuffle(df)



print('Dimensions of the dataframe', df.shape)

print(df[:2])



y = df['label']

X = df.drop(labels=['label'], axis=1).as_matrix()
# reshape to be [samples][pixels][width][height]

print("X:shape", X.shape[0])

X = X.reshape(X.shape[0], 1, 28, 28).astype('float32')

print("X:reshape", X.shape[0])



#normalize inputs 0-255, 0 -1

X = X.astype('float32')

X = X / 255.0



y = np_utils.to_categorical(y)

print(y.shape)

print(y[:5])
from sklearn.metrics import accuracy_score

import pickle

batch_size = 200

n_classes = 10

epochs = 10



def cnn_keras(X, y):

    model = Sequential()

    model.add(Conv2D(32, (5,5), input_shape=(1, 28, 28), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = model

    print(model.summary())

    return model
def train_nn(X, y):

    model = cnn_keras(X, y)

    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    predict = model.predict_classes(X, verbose=0)

    scores = model.evaluate(X, y, verbose=0)

    

    # serialize model to JSON

    mnist_cnn_json = model.to_json()

    with open("mnist_model.json", "w") as json_file:

        json_file.write(mnist_cnn_json)

    # serialize weights to HDF5

    model.save_weights("model.h5")

    print("Saved model to disk")

    

    print("Accuracy: %.2f%%" % (scores[1]*100))

    print(predict[:5])

    return model
train_nn(X, y)
import pandas as pd

df_vald = pd.read_csv("../input/test.csv").as_matrix()



print('Dimensions of the dataframe', df_vald.shape)

#print(df_vald[:1])





#reshape to be samples fixels width height

X_vald = df_vald.reshape(df_vald.shape[0], 1, 28, 28).astype('float32')



#normalize_inputs

X_vald = X_vald/255.0





 



model = cnn_keras(X, y)



prediction = pd.DataFrame()

imageid = []

for i in range(len(X_vald)):

    i = i + 1

    imageid.append(i)

prediction["ImageId"] = imageid 

prediction["Label"] = model.predict_classes(X_vald, verbose=0)

print(prediction[:2])

prediction.to_csv("prediction.csv", index=False)