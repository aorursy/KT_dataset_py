!pip install tensorflow-gpu



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from tensorflow.keras import *

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import  Dense, Activation, Flatten, Dropout, BatchNormalization

# import keras

# from keras import models

# from keras import layers 

from tensorflow.keras.models import Sequential 

# from keras_preprocessing.image import ImageDataGenerator

# from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

# from keras.layers import Conv2D, MaxPooling2D

# from keras import regularizers, optimizers



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from skimage.io import imread



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# # data_train_file = pd.read_csv('train.zip.001/input')

# print(os.listdir("../input/"))

# # shows specific image

# img = mpimg.imread('../input/diabetic-retinopathy-detection/36_right.jpeg')

# # plt.imshow(img)

# # data_train_file = 'train.zip.001'

# # df_train = pd.read_csv(data_train_file)

# # Any results you write to the current directory are saved as output.
import os

# data_train_file = pd.read_csv('train.zip.001/input')

print(os.listdir("../input/"))

# shows specific image

img = mpimg.imread('../input/diabetic-retinopathy-detection/627_right.jpeg')

plt.imshow(img)

# data_train_file = 'train.zip.001'

# df_train = pd.read_csv(data_train_file)

# Any results you write to the current directory are saved as output.
!ls ../input/retinopathy-solution/retinopathy_solution.csv
!ls ../input
# df = pd.trainLabels(np.random.randn(100, 2))



# msk = np.random.rand(len(df)) < 0.8



# train = df[msk]



# In [14]: test = df[~msk]



trainLabels = pd.read_csv('../input/diabetic-retinopathy-detection/trainLabels.csv')



print(len(trainLabels))



# converts elements in "level" column to string so we can use sparse as a parameter flow_rom_dataframe 

trainLabels.level = trainLabels.level.map(lambda x: str(x))



# edits the trainLabels datafram so that elements of the "image" column match case with names of unzipped file

trainLabels.image = trainLabels.image.map(lambda f: f + '.jpeg')

trainLabels

!ls ../lib/kaggle
# creating new dataframe of only unzipped files (had to merge two dataframes together)

from sklearn.model_selection import train_test_split



df_list = os.listdir("../input/diabetic-retinopathy-detection")

df = pd.DataFrame(df_list, columns = ["image"])

df = pd.merge(df, trainLabels, on=['image'], how='inner')

# df['level'].value_counts()

train, test = train_test_split(df, test_size=0.1)

train, validation = train_test_split(train, test_size = 0.2222)



train

train['level'].value_counts()
validation

validation['level'].value_counts()
test

test['level'].value_counts()
datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.4,

    height_shift_range=0.4,

    shear_range=0.4,

    zoom_range=0.4,

    horizontal_flip=True,

    validation_split = .222222)



test_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.4,

    height_shift_range=0.4,

    shear_range=0.4,

    zoom_range=0.4,

    horizontal_flip=True)

# creating batches of augumented and normalized data



train_generator = datagen.flow_from_dataframe(

dataframe = train,

directory = "../input/diabetic-retinopathy-detection",

x_col = "image",

y_col = "level",

subset = 'training',

batch_size = 32,

seed = None,

shuffle = True,

color_mode='rgb',

class_mode = 'sparse',

drop_duplicates = True,

save_prefix = '',

target_size = (28, 28))



validation_generator = datagen.flow_from_dataframe(

dataframe = train,

directory = '../input/diabetic-retinopathy-detection',

x_col = "image",

y_col = "level",

subset = 'validation',

batch_size = 32,

seed = None,

shuffle = True,

color_mode='rgb',

class_mode = 'sparse',

drop_duplicates = True,

save_prefix = '',

target_size = (28, 28))



test_generator = test_datagen.flow_from_dataframe(

dataframe = test,

directory = '../input/diabetic-retinopathy-detection',

x_col = "image",

y_col = "level",

subset = None,

batch_size = 32,

seed = None,

shuffle = True,

color_mode='rgb',

class_mode = 'sparse',

drop_duplicates = True,

save_prefix = '',

target_size = (28, 28))











model = Sequential()



model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 3)))

model.add(BatchNormalization(axis=-1))

convLayer01 = layers.Activation('relu')                    

model.add(convLayer01)



model.add(layers.Conv2D(32, (3, 3)))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))                    

convLayer02 = layers.MaxPooling2D(pool_size=(2,2)) 

model.add(convLayer02)

model.add(Dropout(0.25))





model.add(layers.Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps

model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation

convLayer03 = Activation('relu')                     # activation

model.add(convLayer03)



model.add(layers.Conv2D(64, (3, 3)))

model.add(BatchNormalization(axis=-1))    

model.add(Activation('relu'))  

convLayer04 = layers.MaxPooling2D(pool_size=(2,2)) 

model.add(convLayer04)

model.add(Dropout(0.25))

model.add(layers.Flatten())



model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.003)))

model.add(BatchNormalization())

model.add(Activation('relu'))  

model.add(Dropout(0.5)) 

model.add(layers.Dense(5))

model.add(Activation('softmax')) 

model.summary()




# from keras.utils import to_categorical

# y_binary = to_categorical(train_generator ,num_classes=None, dtype = 'float32') 

model.compile(loss = "sparse_categorical_crossentropy", optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

              metrics =['acc'])



history = model.fit_generator(

    train_generator,

    callbacks=[keras.callbacks.TensorBoard(log_dir='../logs/')],

    validation_data=validation_generator,

    #       steps_per_epoch=100,

      epochs=30)



# model.save('model.h5')

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure

plt.show()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.figure()





plt.show()
model.evaluate_generator(generator=test_generator)
# The predict_classes function outputs the highest probability class

# according to the trained classifier for each input example.

predicted_classes = model.predict_generator(generator = test_generator)

print(predicted_classes)
# ig, m_axs = plt.subplots(2, 4, figsize = (32, 20))

# for (idx, c_ax) in enumerate(m_axs.flatten()):

#     c_ax.imshow(np.clip(test_X[idx]*127+127,0 , 255).astype(np.uint8), cmap = 'bone')

#     c_ax.set_title('Actual Severity: {}\n{}'.format(test_Y_cat[idx], 

#                                                            '\n'.join(['Predicted %02d (%04.1f%%): %s' % (k, 100*v, '*'*int(10*v)) for k, v in sorted(enumerate(pred_Y[idx]), key = lambda x: -1*x[1])])), loc='left')

#     c_ax.axis('off')

# fig.savefig('trained_img_predictions.png', dpi = 300)
!tar -czvf logs.tar ../logs