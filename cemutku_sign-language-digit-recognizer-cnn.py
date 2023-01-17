# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X = np.load('../input/Sign-language-digits-dataset/X.npy')

Y = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(X[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(X[900].reshape(img_size, img_size))

plt.axis('off')
print(X.shape)

print(Y.shape)
# Y Values in [0, 1] range

print(Y.max())

print(Y.min())



# X Values in [0, 1] range

print(X.max())

print(X.min())
# Then lets create x_train, y_train, x_test, y_test arrays

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.20, random_state = 42)



xTrain = xTrain.reshape(-1,64,64,1)

xTest = xTest.reshape(-1,64,64,1)



print(xTrain.shape)

print(yTrain.shape)

print(xTest.shape)

print(yTest.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LearningRateScheduler



model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation = 'relu', input_shape = (64,64,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics = ["accuracy"])
epochs = 30 # 1 epoch means 1 forward and 1 backward pass.

batch_size = 20 # Number of training samples for one forward/backward pass.
datagen = ImageDataGenerator(

        featurewise_center = False,  # set input mean to 0 over the dataset

        samplewise_center = False,  # set each sample mean to 0

        featurewise_std_normalization = False,  # divide inputs by std of the dataset

        samplewise_std_normalization = False,  # divide each input by its std

        zca_whitening = False,  # dimesion reduction

        rotation_range = 10,  # randomly rotate images in the range 10 degrees

        zoom_range = 0.1, # Randomly zoom image 1%

        width_shift_range = 0.1,  # randomly shift images horizontally 1%

        height_shift_range = 0.1,  # randomly shift images vertically 1%

        horizontal_flip = False,  # randomly flip images

        vertical_flip = False)  # randomly flip images



datagen.fit(xTrain)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
# fit the model

history = model.fit_generator(datagen.flow(xTrain,

                                           yTrain, 

                                           batch_size = batch_size), 

                              epochs = epochs, 

                              validation_data = (xTest, yTest), 

                              steps_per_epoch = xTrain.shape[0] // batch_size,

                              callbacks = [annealer])
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color = 'b', label = "validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
print(history.history.keys())
accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

lr = history.history['lr']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label = 'Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()
plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
type(accuracy)

newLr = [x * 100 for x in lr]
plt.plot(epochs, accuracy, 'bo', label = 'Training accuracy')

plt.plot(epochs, newLr, 'b', label = 'Learning Rate')

plt.title('Learning Rate and Accuracy')

plt.legend()

plt.show()
final_loss, final_acc = model.evaluate(xTest, yTest, verbose = 0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# Predict the values from the validation dataset

Y_pred = model.predict(xTest)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(yTest,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()