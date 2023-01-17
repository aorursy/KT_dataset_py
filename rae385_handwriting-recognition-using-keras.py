#packages

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

import pandas as pd



#data processing

mnist_train = pd.read_csv('../input/mnist_train.csv')

mnist_test = pd.read_csv('../input/mnist_test.csv')



train_images = mnist_train.iloc[:, 1:].values

train_labels = mnist_train.iloc[:, :1].values

test_images = mnist_test.iloc[:, 1:].values

test_labels = mnist_test.iloc[:, :1].values



#normalize the data

train_images = train_images.astype('float32')

test_images = test_images.astype('float32')

train_images /= 255   

test_images /= 255



#one hot encoding

train_labels = keras.utils.to_categorical(train_labels, 10)

test_labels = keras.utils.to_categorical(test_labels, 10)
#network topography

model = Sequential()

model.add(Dense(512, activation = 'relu', input_shape=(784,)))

model.add(Dense(10, activation = 'softmax'))

model.summary()
#compiling model and training

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

z = model.fit(train_images, train_labels, 

                   batch_size=100,

                   epochs=10,

                   verbose=2,

                   validation_data=(test_images, test_labels))
score = model.evaluate(test_images, test_labels, verbose=0)

print("Test Accuracy: ", score[1]*100, "%")
#visualize the model working

def predict_test_sample(x):

    label = test_labels[x].argmax(axis=0)

    image = test_images[x].reshape([28,28])

    test_image = test_images[x,:].reshape(1,784)

    prediction = model.predict(test_image).argmax()

    plt.title("Sample %d  Prediction: %d Label: %d" % (x, prediction, label))

    plt.imshow(image, cmap=plt.get_cmap('gray_r'))

    plt.show()
#test the model using the function defined above

for x in range(500):

    image = test_images[x,:].reshape(1,784)

    prediction = model.predict(image).argmax()

    label = test_labels[x].argmax()

    if (prediction != label):

        plt.title("Sample %d  Prediction: %d Label: %d" % (x, prediction, label))

        plt.imshow(image.reshape([28,28]), cmap=plt.get_cmap('gray_r') )

        plt.show()