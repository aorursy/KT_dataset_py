import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
batch_size = 86

num_nets = 15
digits_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

digits_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(digits_train.columns)
X = digits_train.drop(columns="label").values.reshape(digits_train.shape[0],28,28,1) / 255.0

Y = to_categorical(digits_train["label"], num_classes=10)

X_test = digits_test.values.reshape(digits_test.shape[0],28,28,1) / 255.0
plt.figure(figsize=(15, 4.5))

for i in range(30):

    plt.subplot(3, 10, i+1)

    plt.imshow(X[i].reshape((28,28)), cmap=plt.cm.binary)

    plt.axis("off")

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
datagen_train = ImageDataGenerator(

    rotation_range = 10,

    zoom_range = 0.10,

    width_shift_range=0.1,

    height_shift_range=0.1

)
X_ = X[9,].reshape((1,28,28,1))

Y_ = Y[9,].reshape((1,10))





plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    X_generated, Y_generated = datagen_train.flow(X_,Y_).next()

    plt.imshow(X_generated[0].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

    if i==9: X_ = X[11,].reshape((1,28,28,1))

    if i==19: X_ = X[18,].reshape((1,28,28,1))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
model = [0] * num_nets

for j in range(num_nets):

    model[j] = Sequential()

    

    model[j].add(Conv2D(32, kernel_size = 3, activation = "relu", input_shape = (28, 28, 1)))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size = 3, activation = "relu"))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size = 5, strides = 2, padding = "same", activation = "relu"))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))

    

    model[j].add(Conv2D(64, kernel_size = 3, activation = "relu"))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size = 3, activation = "relu"))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size = 5, strides = 2, padding = "same", activation = "relu"))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))

    

    model[j].add(Conv2D(128, kernel_size = 4, activation = "relu"))

    model[j].add(BatchNormalization())

    model[j].add(Flatten())

    model[j].add(Dropout(0.4))

    model[j].add(Dense(10, activation = "softmax"))

    

    

    model[j].compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

history = [0] * num_nets

epochs = 45

for j in range(num_nets):

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.1)

    history[j] = model[j].fit_generator(datagen_train.flow(x_train, y_train, batch_size = 64),

                                       epochs = epochs, 

                                       steps_per_epoch = x_train.shape[0] // 64,

                                       validation_data = (x_val, y_val), 

                                       callbacks = [annealer],

                                       verbose = 0)

    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
# PREDICTIONS AND SUBMIT

results = np.zeros((X_test.shape[0], 10))

for j in range(num_nets):

    results = results + model[j].predict(X_test)

results = np.argmax(results, axis = 1)

results = pd.Series(results, name = "Label")

submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)

submission.ImageId = submission.ImageId.astype(int)

submission.to_csv("prediction.csv", index = False)
plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title(results[i],y=0.95)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()