import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import imageio

import random

import cv2

from pathlib import Path

from keras import optimizers

from keras.optimizers import RMSprop, Adam, SGD

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Activation, SeparableConv2D, Lambda, DepthwiseConv2D, ZeroPadding2D, Input

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, Callback
os.listdir('../input/dice-d4-d6-d8-d10-d12-d20/dice/train')
tr_dir = '../input/dice-d4-d6-d8-d10-d12-d20/dice/train'

val_dir = '../input/dice-d4-d6-d8-d10-d12-d20/dice/valid'




row, col = len(os.listdir(tr_dir)), 10

i = 0

_, ax = plt.subplots(row,col, figsize=(20,15))



for d in os.listdir(tr_dir):

    j = 0

    files  = random.sample(os.listdir(tr_dir +"/"+ d), col)

    for file in files:

        file = tr_dir +"/"+ d + "/" + file

        im=imageio.imread(file)

        ax[i,j].imshow(im,resample=True)

        ax[i,j].set_title(d, fontsize=9)

        j += 1

    i +=1

    

plt.show()



iwidth, iheight = 64, 64

batch_size = 16
Datagen = ImageDataGenerator(

    rotation_range = 360,

    rescale=1. / 255,

    shear_range=0.15,

    zoom_range=0.1,

    #width_shift_range=0.2,

    #height_shift_range=0.2,

    #horizontal_flip=True,

    #vertical_flip = True,

    #fill_mode='nearest',

    #validation_split=0.2

    )
train_generator = Datagen.flow_from_directory(

    tr_dir,

    target_size=(iheight, iwidth),

    batch_size=batch_size,

    seed = 111,

    class_mode='categorical',

    )

validation_generator = Datagen.flow_from_directory(

    val_dir,

    target_size=(iheight, iwidth),

    batch_size=batch_size,

    seed = 111,

    class_mode='categorical',

    )
num_classes = len(train_generator.class_indices.keys())

train_num = train_generator.samples

val_num = validation_generator.samples 

num_classes
minput = Input(shape=(iheight, iwidth, 3))

def addlayers(m, c, d = False):

    m = Conv2D(c, kernel_size=(3, 3), activation ='relu')(m)

    m = Conv2D(c, kernel_size=(5, 5), activation ='relu' , padding='same')(m)

    if d:

        m = DepthwiseConv2D(kernel_size=(3, 3), activation ='relu')(m)

    m = Conv2D(c, kernel_size=(3, 3), activation ='relu')(m)

    m = MaxPooling2D(pool_size = (2, 2))(m)

    m = BatchNormalization()(m)

    return m



bp = ZeroPadding2D((1,1))(minput)



p1 = addlayers(bp, 64)

p2 = addlayers(p1, 126)

p3 = addlayers(p2, 256, True)



fp =  Dropout(0.5)(p3)



fp = Flatten()(fp)



fp = Dense(512, activation = "relu")(fp)

fp = Dropout(0.4)(fp)

fp = Dense(1024, activation = "relu")(fp)

fp = Dropout(0.5)(fp)

fp = Dense(num_classes, activation = "softmax")(fp)



model = Model(inputs=minput, outputs=fp)
model.summary()
model.compile(optimizer=Adam(lr=0.0001),

              loss='binary_crossentropy', 

              metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
filepath=str("dice_project_model.h5f")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# = EarlyStopping(monitor='val_acc', patience=15)

#callbacks_list = [checkpoint]#, stopper]




class EarlyStoppingByAccuracy(Callback):

    def __init__(self, monitor='accuracy', value=1.0, verbose=0):

        super(Callback, self).__init__()

        self.monitor = monitor

        self.value = value

        self.verbose = verbose



    def on_epoch_end(self, epoch, logs={}):

        current = logs.get(self.monitor)

        if current is None:

            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)



        if current >= self.value:

            if self.verbose > 0:

                print("Epoch %05d: early stopping THR" % epoch)

            self.model.stop_training = True

            

            

stopper = EarlyStoppingByAccuracy(monitor='val_acc',value=1.00, verbose=1)







history = model.fit_generator(train_generator,

                              steps_per_epoch = train_num // batch_size,

                              epochs=15,

                              validation_data = validation_generator,

                              validation_steps = val_num // batch_size,

                              callbacks=[learning_rate_reduction, checkpoint, stopper], 

                              verbose = 2

                             )



model.load_weights("dice_project_model.h5f")




f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(history.history['acc'], color = 'blue')

axarr[0].plot(history.history['val_acc'], color = 'purple')

axarr[0].legend(['train', 'test'])

axarr[0].set_title('acc - val_acc')

axarr[1].plot(history.history['loss'], color = 'red')

axarr[1].plot(history.history['val_loss'], color = 'gray')

axarr[1].legend(['train', 'test'])

axarr[1].set_title('loss - val_loss')

plt.show()


