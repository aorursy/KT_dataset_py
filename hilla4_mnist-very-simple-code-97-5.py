import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models

from keras import layers

from keras.utils import to_categorical
train = pd.read_csv("../input/digit-recognizer/train.csv")

X_train = train.drop(labels = ["label"], axis=1)

X_train = X_train.values

X_train = X_train.astype('float32')/255



Y_train = train["label"]

Y_train = Y_train.values

Y_train = to_categorical(Y_train)



X_test = pd.read_csv("../input/digit-recognizer/test.csv")

X_test = X_test.values

X_test = X_test.astype('float32')/255
network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))

network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop', 

                loss = 'categorical_crossentropy',

                metrics = ['accuracy'])
network.fit(X_train, Y_train, epochs=5, batch_size=128)
results = network.predict(X_test)

results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
id = pd.Series(range(1,28001), name="ImageId")
submission = pd.concat([id, results], axis=1)
submission.to_csv("mnist_basic.csv", index=False)