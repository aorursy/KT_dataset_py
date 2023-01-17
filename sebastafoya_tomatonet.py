import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os

import gc
def loadDataset(rootDir):

    

    images = []

    labels = []

    classes = []

    

    for folder in os.listdir(rootDir):

        

        if ('Tomato' not in folder):

            continue

            

        classes.append(folder)

        

        folderPath = os.path.join(rootDir, folder)

        

        imagePaths = [os.path.join(folderPath, file) for file in os.listdir(folderPath) if file.endswith('.JPG')]

        

        for imagePath in imagePaths[:800]:

            

            images.append(cv2.imread(imagePath, cv2.IMREAD_UNCHANGED))

            labels.append(folder)

            

    #Check that all images have the desired size.

    for image in images:

        if (image.shape != (256, 256, 3)):

            cv2.resize(image, (256, 256))

            

    #Normalize image vectors.

    np_images = np.asarray(images, dtype = np.float16) / 255.0

            

    return np_images, labels, classes
#Load data.

images, labels, classes = loadDataset('../input/plantvillage/PlantVillage')



#One-hot the labels.

label_binarizer = LabelBinarizer()

labels = label_binarizer.fit_transform(labels)



#Split into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 42)
print ("number of training examples = " + str(x_train.shape[0]))

print ("number of test examples = " + str(x_test.shape[0]))

print ("X_train shape: " + str(x_train.shape))

print ("Y_train shape: " + str(y_train.shape))

print ("X_test shape: " + str(x_test.shape))

print ("Y_test shape: " + str(y_test.shape))
#Free some memory.

del images

del labels

gc.collect()
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, 

                         zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
nEpochs = 200

learningRate = 0.001

batchSize = 32

width = 256

height = 256

depth = 3

nClasses = len(classes)
model = Sequential()

inputShape = (height, width, depth)

chanDim = -1



if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

    

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))



model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())



model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(nClasses))

model.add(Activation("softmax"))
model.summary()
optimizer = Adam(lr = learningRate, decay = learningRate / nEpochs)



model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size = batchSize),

    validation_data = (x_test, y_test),

    steps_per_epoch = len(x_train) // batchSize,

    epochs=nEpochs, verbose=1

    )
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
#Save the model to disk.

pickle.dump(model, open('tomatoNetModel.pkl', 'wb'))