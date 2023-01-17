import csv

import numpy as np

import tensorflow as tf

import tensorflow.keras.layers as tfl

import pickle

from sklearn.utils import shuffle
def populate_train():

    x_test = []

    y_test = []

    with open("/kaggle/input/Kannada-MNIST/train.csv") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0

        for row in csv_reader:

            if line_count == 0:

                line_count += 1

            else:

                y_test.append(row[0])

                x_test.append(row[1:])

                line_count += 1

        x_test = np.array(x_test)

        x_test = x_test.astype(float)

        y_test = np.array(y_test)

        y_test = y_test.astype(int)

        print(x_test.shape)

        x_test = np.reshape(x_test, (60000, 28, 28, 1))

        x_test, y_test = shuffle(x_test, y_test, random_state=0)

        return x_test, y_test



X, y = populate_train()
def populate_test():

    x_test = []

    y_test = []

    with open("/kaggle/input/Kannada-MNIST/test.csv") as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0

        for row in csv_reader:

            if line_count == 0:

                line_count += 1

            else:

                y_test.append(row[0])

                x_test.append(row[1:])

                line_count += 1

        x_test = np.array(x_test)

        x_test = x_test.astype(float)

        y_test = np.array(y_test)

        y_test = y_test.astype(int)

        print(x_test.shape)

        x_test = np.reshape(x_test, (5000, 28, 28, 1))

        return x_test, y_test



X_test, y_test = populate_test()
model = tf.keras.models.Sequential()



model.add(tfl.Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(tfl.Activation('relu'))

model.add(tfl.MaxPooling2D(pool_size=(2, 2)))



model.add(tfl.Conv2D(256, (3, 3)))

model.add(tfl.Activation('relu'))

model.add(tfl.MaxPooling2D(pool_size=(2, 2)))



model.add(tfl.Flatten())



model.add(tfl.Dense(64))



model.add(tfl.Dense(10))

model.add(tfl.Activation('softmax'))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X, y, batch_size=8, epochs=10)



model.save("Kannada_MNIST_Reader.model")
def prepare(x):

    return x.reshape(-1, 28, 28, 1)
model = tf.keras.models.load_model("Kannada_MNIST_Reader.model")

asserted = [[]]

x = range(5000)

for i in x:

    if (i % 500) == 0:

        print(i)

    predict = np.argmax(model.predict(prepare(X_test[i])))

    asserted.append([i, predict])

asserted.pop(0)



pickle.dump(asserted, open("asserted.pickle", "wb"))
asserted = pickle.load(open("asserted.pickle", "rb"))

variable = ","

np.savetxt("submission.csv", asserted, fmt='%i', delimiter=',', header="id, label", comments='')