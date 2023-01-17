# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Helpe to import cifar-10 data
import pickle as pkl
import glob
import numpy as np

#Unpack tar.gz file
#file = "cifar-10-python.tar.gz"
def unpackFile(file):
    import tarfile
    tar = tarfile.open(file)
    tar.extractall()
    tar.close()
    

def unpickle(fname):
    with open(fname, "rb") as f:
        result = pkl.load(f, encoding='bytes')

    return result


def getData():
    labels_training = []
    dataImgSet_training = []
    labels_test = []
    dataImgSet_test = []

    # use "data_batch_*" for just the training set
    for fname in glob.glob("../input/*data_batch*"):
        #print("Getting data from:", fname)
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            #fname = data[b"filenames"][i]
            labels_training.append(data[b"labels"][i])

            # consecutive 1024 entries store color channels of 32x32 image 
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            
            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  #Change the shape 3,32,32 to 32,32,3 
            dataImgSet_training.append(imgFormat)
            
    # use "test_batch_*" for just the test set
    for fname in glob.glob("../input/*test_batch*"):
        #print("Getting data from:", fname)
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            #fname = data[b"filenames"][i]
            labels_test.append(data[b"labels"][i])

            # consecutive 1024 entries store color channels of 32x32 image 
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            
            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  #Change the shape 3,32,32 to 32,32,3 
            dataImgSet_test.append(imgFormat)
    
    
    dataImgSet_training = np.array(dataImgSet_training)
    labels_training = np.array(labels_training)
    dataImgSet_test = np.array(dataImgSet_test)
    labels_test = np.array(labels_test)
    
    return dataImgSet_training, labels_training, dataImgSet_test, labels_test
#Getting data
#unpackFile("../input/cifar-10-python.tar.gz") Used if the dataset is in the tar.gz format
X_train, y_train, X_test, y_test = getData()

labelNamesBytes = unpickle("../input/batches.meta")
labelNames = []
for name in labelNamesBytes[b'label_names']:
    labelNames.append(name.decode('ascii'))

labelNames = np.array(labelNames)
# Plot CIFAR10 instances
from matplotlib import pyplot
from PIL import Image

# create a grid of 3x3 images
print("Some images")
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(Image.fromarray(X_test[i]))
    #print(labelNames[y_test[i]])
    
pyplot.show()
#Import libs
from time import time
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
model = Sequential()
#32 = número de filtros; kernel de 3x3;
#kernel_constraint=maxnorm(3) = forma de regularização em q os pesos do vetor tenha norma no máximo igual a 3
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
from keras.preprocessing.image import ImageDataGenerator

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(X_train)

#training
batch_size = 64
epochs = 75
lrate = 0.001
#opt_rms = optimizers.rmsprop(lr=0.001,decay=1e-4)
optimizer = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs, verbose=1,
                    validation_data=(X_test,y_test))

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Saving the model
model.save('cifar10_2')