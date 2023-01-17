from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report

from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD

from keras.datasets import cifar10

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
def MiniVGGNet(width, height, depth, classes):

    model = Sequential()

    inputShape = (height, width, depth)

    chanDim = -1



    if K.image_data_format()=="channels_first":

        inputShape = (depth, height, width)

        chanDim = 1



    model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(32, (3,3), padding = "same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding = "same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding = "same"))

    model.add(Activation("relu"))

    model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Activation("relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(classes))

    model.add(Activation("softmax"))

    return model
def step_decay(epoch):

    initAlpha = 0.01

    factor = 0.25

    dropEvery = 5

    

    alpha = initAlpha *(factor **np.floor((1 + epoch)/dropEvery))

    

    return float(alpha)
print("Loading CIFAR-10 data . . . ")

((trainX, trainY), (testX, testY)) = cifar10.load_data()



trainX = trainX.astype("float") / 255.0

testX = testX.astype("float") / 255.0
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print("Compiling Model . . .")

callbacks = [LearningRateScheduler(step_decay)]

opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)

model = MiniVGGNet(width=32, height=32, depth=3, classes=10)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("Training Neural Network . . .")

H = model.fit(trainX, trainY, validation_data=(testX, testY),

batch_size=64, epochs=40, verbose=1)
print("Evaluating Neural network...")

predictions = model.predict(testX, batch_size=64)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
acc = H.history['accuracy']

val_acc = H.history['val_accuracy']

loss = H.history['loss']

val_loss = H.history['val_loss']



epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and Validation accuracy')

plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.figure()