import keras

from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras import optimizers

import numpy as np

from keras.layers.core import Lambda

from keras import backend as K

from keras import regularizers

from keras.utils import np_utils

import os

import numpy as np

import scipy.io

import scipy.misc

import matplotlib.pyplot as plt

import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split
num_classes = 102

weight_decay = 0.0005

x_shape = [32,32,3]
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',

                 input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.summary()
def imread(path):

    img = scipy.misc.imread(path).astype(np.float)

    if len(img.shape) == 2:

        img = np.transpose(np.array([img, img, img]), (2, 0, 1))

    return img

    

path = '../input/101_objectcategories/101_ObjectCategories'

valid_exts = [".jpg", ".gif", ".png", ".jpeg"]

print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))



categories = sorted(os.listdir(path))

ncategories = len(categories)

imgs = []

labels = []

# LOAD ALL IMAGES 

for i, category in enumerate(categories):

    iter = 0

    for f in os.listdir(path + "/" + category):

        if iter == 0:

            ext = os.path.splitext(f)[1]

            if ext.lower() not in valid_exts:

                continue

            fullpath = os.path.join(path + "/" + category, f)

            img = scipy.misc.imresize(imread(fullpath), [32,32, 3])

            img = img.astype('float32')

            img[:,:,0] -= 123.68

            img[:,:,1] -= 116.78

            img[:,:,2] -= 103.94

            imgs.append(img) # NORMALIZE IMAGE 

            label_curr = i

            labels.append(label_curr)

        #iter = (iter+1)%10;

print ("Num imgs: %d" % (len(imgs)))

print ("Num labels: %d" % (len(labels)) )

print (ncategories)
seed = 7

np.random.seed(seed)

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.1)

x_train = np.stack(x_train, axis=0)

y_train = np.stack(y_train, axis=0)

x_test = np.stack(x_test, axis=0)

y_test = np.stack(y_test, axis=0)

print ("Num train_imgs: %d" % (len(x_train)))

print ("Num test_imgs: %d" % (len(x_test)))



y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

#num_classes= y_test.shape[1]
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
#training parameters

batch_size = 128

maxepoches = 300

learning_rate = 0.1

lr_decay = 1e-6

lr_drop = 20
def lr_scheduler(epoch):

    return learning_rate * (0.5 ** (epoch // lr_drop))

reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)





#data augmentation

datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False)  # randomly flip images
datagen.fit(x_train)



#optimization details

sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])



# training process in a for loop with learning rate drop every 25 epoches.

hist = model.fit_generator(datagen.flow(x_train, y_train,

                                 batch_size=batch_size),

                    steps_per_epoch=x_train.shape[0] // batch_size,

                    epochs=maxepoches,

                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

model.save_weights('caltech101-vgg.h5')
scores = model.evaluate(x_test, y_test, verbose=0)

print("Training Accuracy: %.2f%%" % (hist.history['acc'][maxepoches - 1]*100))

print("Testing Accuracy: %.2f%%" % (scores[1]*100))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.legend(['train','test'])

plt.title('loss')

plt.savefig("loss7.png",dpi=300,format="png")

plt.figure()

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.legend(['train','test'])

plt.title('accuracy')

plt.savefig("accuracy7.png",dpi=300,format="png")

model_json = model.to_json()

with open("model7.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model7.h5")

print("Saved model to disk")