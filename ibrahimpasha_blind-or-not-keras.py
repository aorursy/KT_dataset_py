# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2

import random



import os

print(os.listdir("../input"))



from skimage import exposure

from skimage.util import img_as_ubyte

from skimage.color import rgb2gray

from skimage.filters import try_all_threshold

from skimage.filters import threshold_otsu





# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print('Train shape {}\nTest shape{}'.format(train.shape, test.shape))

train.head()
image_list = os.listdir("../input/aptos2019-blindness-detection/train_images")

print(image_list[:5])
def assigncolor(seed):

    random.seed(seed)

    r = random.randint(0, 255)

    g = random.randint(0, 255)

    b = random.randint(0, 255)

    

    return [r, g, b]

def intensity_slicing(grayimage, layers=7):



#     grayimage = img_as_ubyte(grayimage)

#     Global equalize

    grayimage=cv2.addWeighted(grayimage,4, cv2.GaussianBlur( grayimage , (0,0) , 15) ,-4 ,128) 

    

#     grayimage = exposure.equalize_hist(grayimage)

#     grayimage = np.array(grayimage, dtype=np.uint8)

#     grayimage = cv2.cvtColor(grayimage, cv2.COLOR_BGR2GRAY)



#     height = np.size(grayimage, 0)

#     width = np.size(grayimage, 1)





#     colorimage = np.zeros((height, width, 3), np.uint8)



#     n = 256 / layers

#     colormap = list()

#     colormap.append(0)



#     for i in range(layers - 1):

#         colormap.append(int(0 + (i + 1) * n))

#     colormap.append(256)



#     for i in range(height):

#         for j in range(width):

#             for k in range(len(colormap) - 1):

#                 if grayimage[i, j] >= colormap[k] and grayimage[i, j] < colormap[k + 1]:

#                     colorimage[i, j] = assigncolor(k)

    

    grayimage = exposure.equalize_hist(grayimage)

    

#     return grayimage > threshold_otsu(grayimage)

    return grayimage
train['diagnosis'] = train['diagnosis'].astype('str')

train['id_code'] = train['id_code'].map(lambda x: x+'.png')

train_dir = "../input/aptos2019-blindness-detection/train_images/"

index = 59

img =cv2.imread(os.path.join(train_dir, image_list[index]), cv2.IMREAD_UNCHANGED)

img = cv2.resize(img, (300, 300))

img = intensity_slicing(img, 6)

plt.imshow(img.astype('float32'))

train[train.id_code==image_list[index]]
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

#     rescale = 1.0/255,

    validation_split=0.25,

    featurewise_center=True,

    featurewise_std_normalization=True,

#     shear_range=0.2,

#     zoom_range=2.0,

    preprocessing_function=intensity_slicing,

#     rotation_range=20,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

    horizontal_flip=True

)
BATCH_SIZE=32
train_gen=datagen.flow_from_dataframe(

    dataframe=train, 

    directory=train_dir,

    x_col="id_code",

    y_col="diagnosis", 

    class_mode="categorical", 

    target_size=(300,300), 

    batch_size=BATCH_SIZE,

    subset='training',

    shuffle=False

)
plt.imshow(train_gen[0][0][30])

train_gen[0][0][30].shape
valid_gen=datagen.flow_from_dataframe(

    dataframe=train, 

    directory=train_dir,

    x_col="id_code",

    y_col="diagnosis", 

    class_mode="categorical", 

    target_size=(300,300), 

    batch_size=BATCH_SIZE,

    subset='validation',

    shuffle=False

)
test['id_code'] = test['id_code'].map(lambda x: x+'.png')

test_dir = "../input/aptos2019-blindness-detection/test_images/"

test_data_gen = ImageDataGenerator(preprocessing_function=intensity_slicing)

test_generator = test_data_gen.flow_from_dataframe(

    dataframe=test, 

    directory=test_dir,

    x_col="id_code",

    target_size=(300,300), 

    batch_size=BATCH_SIZE,

    shuffle=False,

    class_mode = None

)
plt.imshow(test_generator[0][30])

test_generator[0][20].shape
len(test_generator.filenames)

print(test['id_code'].tail(5), test_generator.filenames[-5:])
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten

from keras.callbacks import ModelCheckpoint, EarlyStopping
model = Sequential()

model.add(Conv2D(16, (3,3), input_shape=(300, 300, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3,3),  activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=2))

# model.add(Conv2D(256, (3,3), activation='relu'))

# model.add(Conv2D(256, (3,3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(5, activation='softmax'))
model.summary()
from keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 5)

checkpoint=ModelCheckpoint('Keras.h5',monitor='val_loss', save_best_only=True)

model.fit_generator(train_gen, epochs=20,

                    steps_per_epoch=len(train_gen.filenames)/BATCH_SIZE,

                    validation_data=valid_gen, 

                    callbacks=[checkpoint, early_stop],

                   validation_steps=len(valid_gen.filenames)/BATCH_SIZE,

                   use_multiprocessing=True)

# from keras.models import load_model



# lmodel = load_model('../input/blind-or-not-keras/Keras.h5')
predicted = model.predict_generator(test_generator, steps=len(test_generator.filenames)/BATCH_SIZE)
test['diagnosis'] = np.argmax(predicted, axis=1)

test['id_code'] = test['id_code'].apply(lambda x: x[:-4])
test.to_csv('submission.csv', index=False)

test.head()