import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten

from tensorflow.keras.models import Sequential, Model

from  matplotlib import pyplot as plt

%matplotlib inline



import hashlib

from imageio import imread

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import os
from distutils.dir_util import copy_tree

fromDirectory = '../input/super-ai-image-classification/train/train/images'

toDirectory = "./image_no_duplicate"

copy_tree(fromDirectory, toDirectory)
img_folder = r'./image_no_duplicate'

csv_file = pd.read_csv(r"../input/super-ai-image-classification/train/train/train.csv",dtype = 'str')
original_cwd = os.getcwd()
original_cwd
os.chdir(r'./image_no_duplicate')

os.getcwd()
file_list = os.listdir()

print(len(file_list))
duplicates = []

hash_keys = dict()

for index, filename in enumerate(os.listdir('.')):

    if os.path.isfile(filename):

        with open(filename,'rb') as f:

            filehash = hashlib.md5(f.read()).hexdigest()

        if filehash not in hash_keys:

            hash_keys[filehash] = index

        else:

            duplicates.append((index,hash_keys[filehash]))



for index in duplicates:

    os.remove(file_list[index[0]])
file_list = os.listdir()

print(len(file_list))
os.chdir(original_cwd)

os.getcwd()
import tensorflow.keras as K

input_t = K.Input(shape = (224,224,3))

res_model = K.applications.ResNet50(include_top = False,

                                   weights = 'imagenet',

                                   input_tensor = input_t)
for layer in res_model.layers[:81]:

    layer.trainable = False
for i, layer in enumerate(res_model.layers):

    print(i, layer.name, '-', layer.trainable)
from keras.preprocessing.image import ImageDataGenerator



BATCH_SIZE = 256

VALIDATION_SPLIT = 0.6

datagen_train = ImageDataGenerator(rescale = 1./255.,

                                    validation_split=VALIDATION_SPLIT,

                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True)



datagen_val = ImageDataGenerator(rescale=1./255, 

                                validation_split=VALIDATION_SPLIT) 



train_generator = datagen_train.flow_from_dataframe(dataframe = csv_file,

                                             directory = img_folder,

                                             x_col = 'id',

                                             y_col = 'category',

                                             batch_size = BATCH_SIZE,

                                             subset="training",

                                             class_mode = 'binary',

                                             target_size = (224,224),

                                             shuffle=True,

                                             seed=0)

val_generator = datagen_val.flow_from_dataframe(dataframe = csv_file,

                                             directory = img_folder,

                                             x_col = 'id',

                                             y_col = 'category',

                                             batch_size = BATCH_SIZE,

                                             subset="validation",

                                             class_mode = 'binary',

                                             target_size = (224,224),

                                             shuffle=True,

                                             seed=0)







# make sure that there is no training file in the validation set

for file in train_generator.filenames:

    if file in val_generator.filenames:

        print('FILE LEAKED!')
from sklearn.utils import class_weight



Y_train = np.array(train_generator.labels)

class_weight = class_weight.compute_class_weight('balanced'

                                               ,np.unique(Y_train)

                                               ,Y_train)

class_weight

class_weight = {0:class_weight[0],1:class_weight[1]}

class_weight
'''

Version 18

Nothing just recommit



Version 17

Nothing just rerun to get the csv

also reduce epoch to 100



Version 16

Actually adding callback

decrease l2 effect from 0.001 to 0.0001

add submission



Version 15

Re enable callback : No I did not.. ,I have to state it on model method too

Add kernel_regularizer=l2(0.001) to Dense





Version 14

reintroduce old data augmentation

                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True

Version13

Unfreeze more layer

from 143 to 81



Version 12

Give back lighter data augmentation

                                    rotation_range=10,

                                    width_shift_range=0.1,

                                    height_shift_range=0.1,

                                    shear_range=0.1,

                                    zoom_range=0.1,

                                    horizontal_flip=True)



Version 11

Drop all of these data augmentation



                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest'



Version 10

DROPOUT 0.25 to 0.5







Version 9

256 BATCH SIZE

SPLIT 0.6/0.4

'''
from keras.regularizers import l2



model = K.models.Sequential()

model.add(res_model)

model.add(K.layers.Flatten())

# model.add(K.layers.BatchNormalization())

# model.add(K.layers.Dense(256,activation = 'relu'))

# # model.add(K.layers.Dropout(0.25))

# model.add(K.layers.BatchNormalization())

# model.add(K.layers.Dense(128,activation = 'relu'))

# # model.add(K.layers.Dropout(0.25))

model.add(K.layers.BatchNormalization())

model.add(K.layers.Dense(64,activation = 'relu',kernel_regularizer=l2(0.0001)))

model.add(K.layers.Dropout(0.5))

model.add(K.layers.BatchNormalization())

model.add(K.layers.Dense(32,activation = 'relu',kernel_regularizer=l2(0.0001)))

model.add(K.layers.Dropout(0.5))

model.add(K.layers.BatchNormalization())

model.add(K.layers.Dense(1,activation = 'sigmoid'))



filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

check_point = K.callbacks.ModelCheckpoint(filepath=filepath,

                                          monitor='val_accuracy',

                                          verbose=1, 

                                          save_best_only=True, 

                                          mode='max')

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),

              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),

              metrics = ['accuracy'])



history = model.fit_generator(train_generator,

                    epochs = 100,

                    verbose = 1,

                   validation_data=val_generator,

                    class_weight = class_weight,

                    callbacks = [check_point])





plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



model.summary()

model.save('model.h5')
test_folder = '../input/super-ai-image-classification/val/val'



test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_directory(directory = test_folder,

                                             target_size=(224, 224),

                                             shuffle = False,

                                             class_mode='categorical',

                                             batch_size=1)



predict = model.predict_generator(test_generator,steps = nb_samples)

predict = predict.flatten()

predict = np.where(predict> 0.5,1,0)



submission = pd.DataFrame({'id':filenames,'category':predict})

submission['id'] = submission['id'].apply(lambda x: x.split('/')[1]) # remove unneccessary suffix

submission.to_csv(r'submission.csv',index = False)