import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import cv2

import pandas as pd

from matplotlib import pyplot as plt

import os

import numpy as np

from sklearn.model_selection import train_test_split

%matplotlib inline
print(os.listdir("../input/grocery-dataset/grocerydataset/"))
shelf_images = "../input/grocery-dataset/grocerydataset/ShelfImages/"

product_images = "../input/grocery-dataset/grocerydataset/ProductImagesFromShelves/"
# lets get shelves photo data from shelf_images



jpg_files = [ f for f in os.listdir(f'{shelf_images}') if f.endswith('JPG') ]

photos_df = pd.DataFrame([[f, f[:6], f[7:14]] for f in jpg_files], columns=['file', 'shelf_id', 'planogram_id'])
print(len(jpg_files))

print(jpg_files[:5])

photos_df.head(5)
# let's get products on shelves photo from ProductImagesFromShelves



products_df = pd.DataFrame([[f[:18], f[:6], f[7:14], i, *map(int, f[19:-4].split('_'))]

                           for i in range(11)

                           for f in os.listdir(f'{product_images}{i}') if f.endswith('png')],

                          columns = ['file', 'shelf_id', 'planogram_id', 'category', 'xmin', 'ymin', 'w', 'h'])



# convert from width, height to xmax, ymax



products_df['xmax'] = products_df['xmin'] + products_df['w']

products_df['ymax'] = products_df['ymin'] + products_df['h']
print(products_df.shape)

(products_df.head(5))
# our data contains many photos of each shelf. In order not to full ourselves, 

# we need to split not by products nor planograms, but by shelves.



# get distinct shelves

shelves = list(set(photos_df['shelf_id'].values))

print(len(shelves))

print(shelves)
# train/test split 

shelves_train, shelves_validation, _, _ = train_test_split(shelves, shelves, test_size=0.3, random_state=42)
# mark all records in dataframes with is_train flag



def is_train(shelf_id):

    return shelf_id in shelves_train



photos_df['is_train'] = photos_df['shelf_id'].apply(is_train)

products_df['is_train'] = products_df['shelf_id'].apply(is_train)
photos_df.head(5)
products_df.head(5)
df = products_df[products_df.category != 0].groupby(['category', 'is_train'])['category'].count().unstack('is_train').fillna(0)



df.plot(kind='barh', stacked=True)

plt.show()
# fuction to display shelf photos with rectangle product



def draw_shelf_photo(file):

    file_products_df = products_df[products_df.file == file]

    coordinates = file_products_df[['xmin','ymin','xmax','ymax']].values

    im = cv2.imread(f'{shelf_images}{file}')

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    for xmin,ymin,xmax,ymax in coordinates:

        cv2.rectangle(im, (xmin,ymin), (xmax,ymax), (0,255,0), 5)

    plt.imshow(im)
# draw one photo to check our data



fig = plt.gcf()

fig.set_size_inches(12, 6)

draw_shelf_photo('C1_P03_N1_S2_1.JPG')
from IPython.display import Image

Image('../input/grocery-dataset-extra-images/brands.png', width=500)
# They proposed approach to brands recognition as a combination of following algorithms.

Image('../input/grocery-dataset-extra-images/brand_recognition.png', width=500)
# this approach accuracy is below

Image('../input/grocery-dataset-extra-images/brand_recognition_accuracy.png', width=500)
import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import itertools



from sklearn.metrics import confusion_matrix



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten, Input, Activation

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import BatchNormalization



from keras.regularizers import l2

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler

from keras import backend as K



%matplotlib inline
# we already have photos_df and products_df ready



# neural networks work with input of fixed size, so we need to resize our

# packs images to the chosen size. The size is some kind of metaparameter and 

# you should try different variants. Logically, the bigger size you select,

# the better performace you'll have. Unfortunatelly it is not true, because 

# of over fitting. The more parameters your neural network have, the easier it

# became over fitted



num_classes = 10

SHAPE_WIDTH = 80

SHAPE_HEIGHT = 120
# resize pack to fixed size SHAPE_WIDTH*SHAPE_HEIGHT



def resize_pack(pack):

    fx_ratio = SHAPE_WIDTH / pack.shape[1]

    fy_ratio = SHAPE_HEIGHT / pack.shape[0]

    pack = cv2.resize(pack, (0,0), fx=fx_ratio, fy=fy_ratio)

    return pack[0:SHAPE_HEIGHT, 0:SHAPE_WIDTH]
print(photos_df.columns)

print(products_df.columns)
# x - image

# y - class

# f - is_train flag

x, y, f = [], [], []



for file, is_train in photos_df[['file','is_train']].values:

    photos_rects = products_df[products_df['file'] == file]

    rects_data = photos_rects[['category','xmin','ymin','xmax','ymax']]

    im = cv2.imread(f'{shelf_images}{file}')

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    for category, xmin, ymin, xmax, ymax in rects_data.values:

        if category == 0:

            continue

        pack = resize_pack(np.array(im[ymin:ymax, xmin:xmax]))

        x.append(pack)

        f.append(is_train)

        y.append(category - 1)
# display one SHAPE_WIDTH x SHAPE_HEIGHT resized pack image, 

# it is hard to recognize category with our eyes, let's see

# how neural network will do the job

plt.imshow(x[60])

plt.show()
# lets split the data into train/validation dataset based on is_train flag

x,y,f = np.array(x), np.array(y), np.array(f)

x_train, x_validation, y_train, y_validation = x[f], x[~f], y[f], y[~f]



# save validation images

x_validation_images = x_validation
x_train.shape, x_validation.shape, y_train.shape, y_validation.shape
# convert y_train and y_validation into one hot arrays

y_train = keras.utils.to_categorical(y_train, num_classes)

y_validation = keras.utils.to_categorical(y_validation, num_classes)
y_train.shape, y_validation.shape
# normalize x_train and x_validation

x_train, x_validation = x_train.astype('float32'), x_validation.astype('float32')

x_train /= 255

x_validation /= 255
# let's see what do we have

print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)

print(x_train.shape[0], 'train samples')

print(x_validation.shape[0], 'validation samples')
def lr_schedule(epoch):

    lr = 1e-3

    if epoch > 5:

        lr *= 1e-1

    print("learning rate: ", lr)

    return lr
def resnet_layer(inputs,

                 num_filters=16,

                 kernel_size=3,

                 strides=1,

                 activation='relu',

                 batch_normalization=True,

                 conv_first=True):

    conv = Conv2D(num_filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-4))



    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

    return x
def resnet_v1(input_shape, depth, num_classes=10):

    if (depth - 2) % 6 != 0:

        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.

    num_filters = 16

    num_res_blocks = int((depth - 2) / 6)



    inputs = Input(shape=x_train.shape[1:])

    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack

                strides = 2  # downsample

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])

            x = Activation('relu')(x)

        num_filters *= 2



    # Add classifier on top.

    # v1 does not use BN after last shortcut connection-ReLU

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)



    # Instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model



n = 3

version = 1

if version == 1:

    depth = n * 6 + 2

elif version == 2:

    depth = n * 9 + 2

model_type = 'ResNet%dv%d' % (depth, version)



model = resnet_v1(input_shape=x_train.shape[1:], depth=depth, num_classes=num_classes)

model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
# lets see our model architecture and how many trainable parammeters it has

#model.summary()
# this will do preprocessing and reall time data augmentation



datagen = ImageDataGenerator(featurewise_center = False, # set input mean to 0 over the dataset

                             samplewise_center = False, # set each sample mean to 0

                             featurewise_std_normalization = False, # divide inputs by the std of the dataset

                             samplewise_std_normalization = False, # divide each input by its std

                             zca_whitening = False, # apply zca whitening

                             rotation_range = 5, # randomly rotate images in the range (degrees, 0 to 180)

                             width_shift_range = 0.1, # randomly shift images horizontally (fraction of total width)

                             height_shift_range = 0.1, # randomly shift images vertically (fraction of total height)

                             horizontal_flip = False, # randomly flip images

                             vertical_flip = False) # randomly flip images      

datagen.fit(x_train)
# let's run training process, 20 epochs is enough

batch_size = 50

epochs = 15

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                    validation_data=(x_validation, y_validation),

                    epochs=epochs, verbose=1, workers=4, 

                    callbacks=[LearningRateScheduler(lr_schedule)])