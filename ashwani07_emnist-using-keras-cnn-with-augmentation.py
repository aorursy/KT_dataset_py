# Import modules

# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

import cv2



#keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

import sklearn.metrics as metrics

from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("../input/emnist-balanced-train.csv",delimiter = ',')

test = pd.read_csv("../input/emnist-balanced-test.csv", delimiter = ',')

mapp = pd.read_csv("../input/emnist-balanced-mapping.txt", delimiter = ' ', \

                   index_col=0, header=None, squeeze=True)

print("Train: %s, Test: %s, Map: %s" %(train.shape, test.shape, mapp.shape))
# Constants

HEIGHT = 28

WIDTH = 28
# Split x and y

train_x = train.iloc[:,1:]

train_y = train.iloc[:,0]

del train



test_x = test.iloc[:,1:]

test_y = test.iloc[:,0]

del test
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
def rotate(image):

    image = image.reshape([HEIGHT, WIDTH])

    image = np.fliplr(image)

    image = np.rot90(image)

    return image
# Flip and rotate image

train_x = np.asarray(train_x)

train_x = np.apply_along_axis(rotate, 1, train_x)

print ("train_x:",train_x.shape)



test_x = np.asarray(test_x)

test_x = np.apply_along_axis(rotate, 1, test_x)

print ("test_x:",test_x.shape)
# Normalise

train_x = train_x.astype('float32')

train_x /= 255

test_x = test_x.astype('float32')

test_x /= 255

# plot image

for i in range(100, 109):

    plt.subplot(330 + (i+1))

    plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))

    plt.title(chr(mapp[train_y[i]]))
# number of classes

num_classes = train_y.nunique()
# One hot encoding

train_y = np_utils.to_categorical(train_y, num_classes)

test_y = np_utils.to_categorical(test_y, num_classes)

print("train_y: ", train_y.shape)

print("test_y: ", test_y.shape)
# Reshape image for CNN

train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)

test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)
# partition to train and val

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size= 0.10, random_state=7)
# Building model

# ((Si - Fi + 2P)/S) + 1

model = Sequential()



model.add(Conv2D(filters=128, kernel_size=(5,5), padding = 'same', activation='relu',\

                 input_shape=(HEIGHT, WIDTH,1)))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3) , padding = 'same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(.5))

model.add(Dense(units=num_classes, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# create an image generator and batches

# datagen = ImageDataGenerator(rotation_range=60, zoom_range=(0.9, 1.1), \

#                             width_shift_range=0.2, height_shift_range=0.2)

datagen = ImageDataGenerator()

datagen.fit(train_x)

train_gen = datagen.flow(train_x, train_y, batch_size=32)
# train the model and get performance data

history = model.fit_generator(train_gen, steps_per_epoch=len(train_x)/32, epochs=10, \

                             validation_data=(val_x, val_y),)
# plot accuracy and loss

def plotgraph(epochs, acc, val_acc):

    # Plot training & validation accuracy values

    plt.plot(epochs, acc, 'b')

    plt.plot(epochs, val_acc, 'r')

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()
#%%

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
# Accuracy curve

plotgraph(epochs, acc, val_acc)
# loss curve

plotgraph(epochs, loss, val_loss)
score = model.evaluate(test_x, test_y, verbose=0)

print("Test loss:", score[0])

print("Test accuracy:", score[1])
y_pred = model.predict(test_x)

y_pred = (y_pred > 0.5)
cm = metrics.confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))