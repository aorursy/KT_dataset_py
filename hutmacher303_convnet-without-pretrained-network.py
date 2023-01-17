# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.image import imread

import seaborn as sns

%matplotlib inline



# - Sklearn

from sklearn.model_selection import train_test_split



# - Keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.utils import np_utils

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from keras.optimizers import Adam, RMSprop





import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# def read_images(img_paths, img_height=image_size, img_width=image_size):

#     imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

#     img_array = np.array([img_to_array(img) for img in imgs])

#     return(img_array)



def read_pix(jpg_dir):

    filenames = glob.glob(os.path.join(jpg_dir, '*.jpg'))

    img_array = np.zeros((len(filenames), 32, 32, 3), dtype=int)  # images as np.array

    img_index = []  # image filenames in correct order

    for idx, filename in enumerate(filenames):

        im_tmp = matplotlib.image.imread(filename)

        img_array[idx, :, :, :] = np.array(im_tmp, dtype=int)

        img_index.append(os.path.basename(filename))

    return img_array, img_index



def prepare_data(img_array, img_index, train_response, test_size=0.25):

    

    # - Response

    y_train_series, y_test_series = train_test_split(train_response, 

                                                     shuffle=True,

                                                     random_state=12,

                                                     test_size=test_size)

    y_train = y_train_series.values[:, np.newaxis]

    y_test = y_test_series.values[:, np.newaxis]



    # - Predictors

    train_index = [img_index.index(idx) for idx in y_train_series.index]

    test_index = [img_index.index(idx) for idx in y_test_series.index]

    x_train = img_array[train_index, :, :, :]

    x_test = img_array[test_index, :, :, :]



    return x_train, y_train, x_test, y_test, y_train_series, y_test_series



def simple_oversample(x, y, oversample_class, oversample_factor=3):

    new_x = x.copy()

    new_y = y.copy()

    oversample_mask = np.ravel(y == oversample_class)

    oversample_x = x[oversample_mask, :, :, :]

    oversample_y = y[oversample_mask, :]

    for i in range(oversample_factor):

        new_x = np.concatenate([new_x, oversample_x], axis=0)

        new_y = np.concatenate([new_y, oversample_y], axis=0)

    return new_x, new_y
train_labels = pd.read_csv('../input/train.csv')  # image labels in training data

sample_submission = pd.read_csv('../input/sample_submission.csv')  # example submission



# - Read training data (images as np.arrays, associated labels as list)

train_img_array, train_img_index = read_pix('../input/train/train')

test_img_array, test_img_index = read_pix('../input/test/test')

    

# - Put labels into pd.Series object

train_response = train_labels['has_cactus']

train_response.index = train_labels['id']

train_response = train_response.loc[train_img_index]
# - Image labels

print('Image Labels')

print(train_labels.head(2))

print('\n')



# - Image labels

print('Image Labels (Series)')

print(train_response.head(2))

print('\n')



# - Submission format

print('Submission Example')

print(sample_submission.head(2))
print('Some examples')



np.random.seed(27)

inspect = np.random.randint(low=0, high=train_img_array.shape[0], size=9)

# inspect = np.arange(9)

for i in range(9):

    pic_id = inspect[i]

    plt.subplot(330 + 1 + i)

    plt.imshow(train_img_array[pic_id].astype(int))

plt.show()



print('Image Labels')

train_response.loc[np.array(train_img_index)[inspect]]
count_classes = pd.crosstab(train_response, columns='count')

count_classes.index = ['no cactus', 'cactus']

count_classes.plot(kind='bar', legend=False)

plt.xticks(rotation=0)

plt.title('Number of instances in training data')

plt.show()

ratio = count_classes.loc['cactus', 'count'] / count_classes.loc['no cactus', 'count']

print('Ratio (cactus vs. no cactus): {:.2f}'.format(ratio))
# - Split into training and validation set (25% validation)

x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(train_img_array, train_img_index, train_response)



count_classes_train = pd.crosstab(y_train_series, columns='count')

count_classes_train.index = ['no cactus', 'cactus']

count_classes_train.plot(kind='bar', legend=None)

plt.xticks(rotation=0)

plt.title('Number of instances in train data')

plt.show()

ratio = count_classes_train.loc['cactus', 'count'] / count_classes_train.loc['no cactus', 'count']

print('Ratio (cactus vs. no cactus): {:.2f}'.format(ratio))
print('Some images from the training set')



np.random.seed(26)

inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)

# inspect = np.arange(9)

for i in range(9):

    pic_id = inspect[i]

    plt.subplot(330 + 1 + i)

    plt.imshow(x_train[pic_id].astype(int))

plt.show()



print('Accompanying labels')



y_train[inspect]
# - Parameters

batch_size = 128

num_classes = 2



# Generators

datagen = ImageDataGenerator(

    rotation_range=90,

    width_shift_range=.1,

    height_shift_range=.1,

    shear_range=.2,

    zoom_range=.1,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest',

)



# - Prepare and oversample data

x_train, y_train, x_test, y_test, y_train_series, y_test_series = prepare_data(train_img_array, train_img_index, train_response, test_size=0.2)

x_train, y_train = simple_oversample(x_train, y_train, oversample_class=0, oversample_factor=2)

x_test, y_test = simple_oversample(x_test, y_test, oversample_class=0, oversample_factor=2)



print("Number of 'no cactus' samples in y_train: {}".format((y_train == 0).sum()))

print("Number of 'cactus' samples in y_train: {}".format((y_train == 1).sum()))
print('Some images from the oversampled training set')



np.random.seed(26)

inspect = np.random.randint(low=0, high=y_train.shape[0], size=9)

# inspect = np.arange(9)

for i in range(9):

    pic_id = inspect[i]

    plt.subplot(330 + 1 + i)

    plt.imshow(x_train[pic_id].astype(int))

plt.show()



print('Accompanying labels')



y_train[inspect]
# Normalize inputs

x_train = x_train / 255

x_test = x_test / 255



# # One hot encoding of response

# y_train = np_utils.to_categorical(y_train, num_classes=num_classes)

# y_test = np_utils.to_categorical(y_test, num_classes=num_classes)



train_generator = datagen.flow(x_train, y_train)

test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow(x_test, y_test)
def convnet_model():

    # create model

    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation='relu'))

    model.add(Conv2D(32, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # compile model

#     model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-6), metrics=['accuracy'])

    return model
# build model

model = convnet_model()

# fit model

# hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=200, verbose=2)

hist = model.fit_generator(train_generator,

                           steps_per_epoch=np.ceil(x_train.shape[0] / 32),

                           epochs=120,

                           validation_data=validation_generator,

                           validation_steps=np.ceil(x_test.shape[0] / 32)

                   )

# final evaluation

scores = model.evaluate(x_test, y_test, verbose=0)

print('CNN error: {:.2f}'.format(100-scores[1]*100))
# summarize history for accuracy

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
probability = model.predict_proba(test_img_array/255)

res = pd.DataFrame({

    'id': test_img_index,

    'has_cactus': probability.ravel(),

})

res.to_csv('submission.csv', index=False)
res.head()