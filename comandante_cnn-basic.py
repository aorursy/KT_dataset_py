# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory









# Any results you write to the current directory are saved as output.
import keras

keras.__version__

import numpy as np # forlinear algebra

import matplotlib.pyplot as plt #for plotting things

import os

from PIL import Image

print(os.listdir("../input"))



# Keras Libraries

import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D,BatchNormalization

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img





mainDIR = os.listdir('../input/dataset/dataset')

print(mainDIR)
train_folder= '../input/dataset/dataset/train/'

test_folder = '../input/dataset/dataset/validation/'

val_folder = '../input/dataset/dataset/test/'

# train 

os.listdir(train_folder)

train_kalem = train_folder+'kalem/'

train_kumanda = train_folder+'kumanda/'
#Kalem pic 

rand_norm= np.random.randint(0,len(os.listdir(train_kalem)))

norm_pic = os.listdir(train_kalem)[rand_norm]

print('kalem picture title: ',norm_pic)



norm_pic_address = train_kalem+norm_pic



#Kumanda

rand_p = np.random.randint(0,len(os.listdir(train_kumanda)))



sic_pic =  os.listdir(train_kumanda)[rand_norm]

sic_address = train_kumanda+sic_pic

print('kumanda picture title:', sic_pic)



# Load the images

norm_load = Image.open(norm_pic_address)

sic_load = Image.open(sic_address)



#Let's plt these images

f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(norm_load)

a1.set_title('kalem')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title('kumanda')
print('total training kalem images:', len(os.listdir(train_kalem)))
print('total training kumanda images:', len(os.listdir(train_kumanda)))
cnn = Sequential()





cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

cnn.add(Conv2D(32, (3, 3), activation="relu"))

cnn.add(BatchNormalization())

cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Conv2D(64, (3, 3), activation="relu"))

cnn.add(Conv2D(64, (3, 3), activation="relu"))

cnn.add(BatchNormalization())

cnn.add(MaxPooling2D(pool_size = (2, 2)))





# Flatten the layer

cnn.add(Flatten())



# Fully Connected Layers

cnn.add(Dense(activation = 'relu', units = 1024))

cnn.add(BatchNormalization())

cnn.add(Dense(activation = 'sigmoid', units = 1))
cnn.summary()
#Compile the model

from keras import optimizers

cnn.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.001),

              metrics=['accuracy'])
# Fitting the CNN to the images

# # All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale = 1./255)



test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/train',

                                                 target_size = (64, 64),

                                                 batch_size = 5,

                                                 class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/dataset/dataset/test/',

    target_size=(64, 64),

    batch_size=5,

    class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/dataset/dataset/validation',

                                            target_size = (64, 64),

                                            batch_size = 5,

                                            class_mode = 'binary')
#Fitting our model

#steps_per_epoch=total training data/batch_size

cnn_model = cnn.fit_generator(training_set,

                         steps_per_epoch = 12,

                         epochs = 20,

                         validation_data = validation_generator,

                         validation_steps = 4)
cnn.save('kalem_kumanda_small_1.h5')
plt.plot(cnn_model.history['acc'])

plt.plot(cnn_model.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train acc', 'validation acc'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_model.history['loss'])

plt.plot(cnn_model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train loss', 'validation loss'], loc='upper left')

plt.show()
cnn_reduce = Sequential()





cnn_reduce.add(Conv2D(4, (3, 3), activation="relu", input_shape=(64, 64, 3)))

cnn_reduce.add(Conv2D(16, (3, 3), activation="relu"))

cnn_reduce.add(BatchNormalization())

cnn_reduce.add(MaxPooling2D(pool_size = (2, 2)))







# Flatten the layer

cnn_reduce.add(Flatten())



# Fully Connected Layers

cnn_reduce.add(Dense(activation = 'relu', units = 512))

cnn_reduce.add(BatchNormalization())

cnn_reduce.add(Dense(activation = 'sigmoid', units = 1))
#Compile the model

from keras import optimizers

cnn_reduce.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.001),

              metrics=['accuracy'])
# Fitting the CNN to the images

# # All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale = 1./255)



test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/train',

                                                 target_size = (64, 64),

                                                 batch_size = 5,

                                                 class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/dataset/dataset/test/',

    target_size=(64, 64),

    batch_size=5,

    class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/dataset/dataset/validation',

                                            target_size = (64, 64),

                                            batch_size = 5,

                                            class_mode = 'binary')
#Fitting our model

#steps_per_epoch=total training data/batch_size

cnn_reduce_model = cnn.fit_generator(training_set,

                         steps_per_epoch = 12,

                         epochs = 20,

                         validation_data = validation_generator,

                         validation_steps = 4)
cnn_reduce.save('kalem_kumanda_small_reduce.h5')
plt.plot(cnn_reduce_model.history['acc'])

plt.plot(cnn_reduce_model.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train acc', 'validation acc'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_reduce_model.history['loss'])

plt.plot(cnn_reduce_model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train loss', 'validation loss'], loc='upper left')

plt.show()
# summarize history for validation loss 

plt.plot(cnn_model.history['val_loss'])

plt.plot(cnn_reduce_model.history['val_loss'])

plt.title(' validation model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['cnn_model', 'cnn_reduce_model'], loc='upper left')

plt.show()
from keras import regularizers

cnn_regular = Sequential()





cnn_regular.add(Conv2D(32, (3, 3), activation="relu",kernel_initializer='he_normal', input_shape=(64, 64, 3)))

cnn_regular.add(Conv2D(32, (3, 3), activation="relu",kernel_regularizer=regularizers.l2(0.01)))

cnn_regular.add(BatchNormalization())

cnn_regular.add(MaxPooling2D(pool_size = (2, 2)))

cnn_regular.add(Conv2D(64, (3, 3), activation="relu",kernel_regularizer=regularizers.l2(0.01)))

cnn_regular.add(Conv2D(64, (3, 3), activation="relu",kernel_regularizer=regularizers.l2(0.01)))

cnn_regular.add(BatchNormalization())

cnn_regular.add(MaxPooling2D(pool_size = (2, 2)))





# Flatten the layer

cnn_regular.add(Flatten())



# Fully Connected Layers

cnn_regular.add(Dense(activation = 'relu', units = 1024,kernel_initializer='he_normal'))

cnn_regular.add(BatchNormalization())

cnn_regular.add(Dense(activation = 'sigmoid', units = 1))
#Compile the model

from keras import optimizers

cnn_regular.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.001),

              metrics=['accuracy'])
# Fitting the CNN to the images

# # All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale = 1./255)



test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/train',

                                                 target_size = (64, 64),

                                                 batch_size = 5,

                                                 class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/dataset/dataset/test/',

    target_size=(64, 64),

    batch_size=5,

    class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/dataset/dataset/validation',

                                            target_size = (64, 64),

                                            batch_size = 5,

                                            class_mode = 'binary')
#Fitting our model

#steps_per_epoch=total training data/batch_size

cnn_regular_model = cnn_regular.fit_generator(training_set,

                         steps_per_epoch = 12,

                         epochs = 20,

                         validation_data = validation_generator,

                         validation_steps = 4)
plt.plot(cnn_regular_model.history['acc'])

plt.plot(cnn_regular_model.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train acc', 'validation acc'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_regular_model.history['loss'])

plt.plot(cnn_regular_model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train loss', 'validation loss'], loc='upper left')

plt.show()
# summarize history for validation loss 

plt.plot(cnn_model.history['val_loss'])

plt.plot(cnn_regular_model.history['val_loss'])

plt.title(' validation model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['cnn_model', 'cnn_reduce_model'], loc='upper left')

plt.show()
from keras.layers import Dropout

cnn_data = Sequential()





cnn_data.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

cnn_data.add(Conv2D(32, (3, 3), activation="relu"))

cnn_data.add(BatchNormalization())

cnn_data.add(MaxPooling2D(pool_size = (2, 2)))

cnn_data.add(Conv2D(64, (3, 3), activation="relu"))

cnn_data.add(Conv2D(64, (3, 3), activation="relu"))

cnn_data.add(BatchNormalization())

cnn_data.add(MaxPooling2D(pool_size = (2, 2)))





# Flatten the layer

cnn_data.add(Flatten())

cnn_data.add(Dropout(rate=0.5))



# Fully Connected Layers

cnn_data.add(Dense(activation = 'relu', units = 1024))

cnn_data.add(BatchNormalization())

cnn_data.add(Dense(activation = 'sigmoid', units = 1))
#Compile the model

from keras import optimizers

cnn_data.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=0.001),

              metrics=['accuracy'])
# Fitting the CNN to the images

# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

#Data Augmentation



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/train',

                                                 target_size = (64, 64),

                                                 batch_size = 5,

                                                 class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/dataset/dataset/validation/',

    target_size=(64, 64),

    batch_size=5,

    class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/dataset/dataset/test',

                                            target_size = (64, 64),

                                            batch_size = 5,

                                            class_mode = 'binary')

#Fitting our model

#steps_per_epoch=total training data/batch_size

cnn_data_model = cnn_data.fit_generator(training_set,

                         steps_per_epoch = 12,

                         epochs = 20,

                         validation_data = validation_generator,

                         validation_steps = 4)
plt.plot(cnn_data_model.history['acc'])

plt.plot(cnn_data_model.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train acc', 'validation acc'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_data_model.history['loss'])

plt.plot(cnn_data_model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train loss', 'validation loss'], loc='upper left')

plt.show()