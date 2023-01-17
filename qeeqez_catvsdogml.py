import numpy as np # linear algebra

import random

import json

import csv



from matplotlib import pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing import image

from keras import optimizers



import os
!rm -rf ../train

!rm -rf ../test

!unzip -oj ../input/dogs-vs-cats/train.zip -d ../train/

!unzip -oj ../input/dogs-vs-cats/test1.zip -d ../test/
TRAIN_DIR = '../train/'

TEST_DIR = '../test/'



ROWS = 150

COLS = 150

CHANNELS = 3



BATCH_SIZE=128

EPOCHS=10
original_train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]



# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

original_train_images = train_dogs[:12000] + train_cats[:12000]

random.shuffle(original_train_images)

# test_images =  test_images[:100]



# section = int(len(original_train_images) * 0.8)

train_images = original_train_images[:18000]

validation_images = original_train_images[18000:]
def plot_arr(arr):

    plt.figure()

    plt.imshow(image.array_to_img(arr))

    plt.show()



def plot(img):

    plt.figure()

    plt.imshow(img)

    plt.show()
def prep_data(images):

    count = len(images)

    X = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.float32)

    y = np.zeros((count,), dtype=np.float32)

    

    for i, image_file in enumerate(images):

        img = image.load_img(image_file, target_size=(ROWS, COLS))

        X[i] = image.img_to_array(img)

        if 'dog' in image_file:

            y[i] = 1.

        if i%1000 == 0: print('Processed {} of {}'.format(i, count))

    

    return X, y
X_train, y_train = prep_data(train_images)

X_validation, y_validation = prep_data(validation_images)

X_test, y_test = prep_data(test_images)
print(f"Total train: {len(X_train)}; Total test: {len(X_validation)}; Total test: {len(X_test)}")
train_datagen = image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)



validation_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(

    X_train,

    y_train,

    batch_size=BATCH_SIZE)



validation_generator = validation_datagen.flow(

    X_validation,

    y_validation,

    batch_size=BATCH_SIZE)
X_tmp = np.array(X_train[100:101])

y_tmp = np.array(y_train[100:101])



example_generator = train_datagen.flow(

    X_tmp, 

    y_tmp,

    batch_size=BATCH_SIZE

)



plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
from keras.applications import VGG16

from keras.models import Model



def create_vgg16():

    pre_trained_model = VGG16(input_shape=(ROWS, COLS, CHANNELS), include_top=False, weights="imagenet")



    for layer in pre_trained_model.layers[:15]:

        layer.trainable = False



    for layer in pre_trained_model.layers[15:]:

        layer.trainable = True



    last_layer = pre_trained_model.get_layer('block5_pool')

    last_output = last_layer.output



    x = MaxPooling2D()(last_output)

    x = Flatten()(x)

    x = Dropout(0.25)(x)

    x = Dense(512, activation='relu')(x)

    x = Dense(1, activation='sigmoid')(x)

    

    vgg16model = Model(pre_trained_model.input, x)

    

    return vgg16model
vgg16model = create_vgg16()

vgg16model.summary()
vgg16model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])
train_steps = len(train_images)/BATCH_SIZE

validation_steps = len(validation_images)/BATCH_SIZE



history = vgg16model.fit(

    X_train, 

    y_train, 

    epochs=EPOCHS, 

    batch_size=128, 

    verbose=2, 

    validation_data=(X_validation, y_validation)

)
import pandas as pd

test_generator = validation_datagen.flow(

    X_test,

    y_test,

    batch_size=BATCH_SIZE)





test_filenames = os.listdir("../test/")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]

nb_samples
predict = vgg16model.predict(X_test)

threshold = 0.5

test_df['category'] = np.where(predict > threshold, 1,0)
import seaborn as sns

submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)



plt.figure(figsize=(10,5))

sns.countplot(submission_df['label'])

plt.title("(Test data)")
os.chdir("/kaggle/working/")

submission_df.to_csv('submission.csv', index=False)