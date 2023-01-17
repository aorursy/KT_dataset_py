#imports

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D

from tensorflow.keras.optimizers import Adam

import random
#image shape is defined for what we will feed into keras model

IMAGE_SHAPE = (110, 110, 1)



# will feed through data set via mapping

def load_and_preprocess_image(image):

    image = tf.image.resize(image, IMAGE_SHAPE[0:2])

    image = tf.cast(image, tf.float64)

    image /= 255.0  # normalize to [0,1] range

    return image



# augmentations will feed through training data set only

def random_bright(image):

    return tf.image.random_brightness(image, 0.1)

            

def random_contrast(image):

    return tf.image.random_contrast(image, 0.9, 1.1)



# augmentation includes flips and bright/contrast changes

# there still isn't tensorflow warping as a built in function

def augment_image(image, label):

    if (label == 1 and random.random() < 0.8) or (label == 0 and random.random() < 0.3):

        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_flip_up_down(image)

        image = random_contrast(image)

        image = random_bright(image)

    return image, label
#read in dataframes

x_train_df = pd.read_csv('../input/volcanoes_train/train_images.csv', header=None)

y_train_df = pd.read_csv('../input/volcanoes_train/train_labels.csv')
# from https://www.kaggle.com/behcetsenturk/finding-volcanoes-with-cnn

def corruptedImages(data):

    corruptedImagesIndex = []

    for index, image in enumerate(np.resize(data, (data.shape[0], 12100))): # resize (7000, 110, 110, 1) to (7000,12100)

        sum = 0;

        for pixelIndex in range(0,len(image)):

            sum += image[pixelIndex]

            if pixelIndex == 10:

                break

        if sum == 0:

            corruptedImagesIndex.append(index)

        else:

            sum = 0



    for index, image in enumerate(np.resize(data, (data.shape[0], 12100))): # resize (7000, 110, 110, 1) to (7000,12100)

        sum = 0;

        for pixelIndex in range(0,len(image),110):

            sum += image[pixelIndex]

            if pixelIndex == 10:

                break

        if sum == 0 and index not in corruptedImagesIndex:

            corruptedImagesIndex.append(index)

        else:

            sum = 0

    return corruptedImagesIndex



corrupted_indexes = corruptedImages(x_train_df)

x_train_df = x_train_df.drop(corrupted_indexes).reset_index(drop=True)

y_train_df = y_train_df.drop(corrupted_indexes).reset_index(drop=True)
# look at distribution, we'll see there are 6 times volcano = 0 vs volcano = 1

y_train_df.groupby('Volcano?')['Volcano?'].count().plot.bar()
# initial data set size

DS_SIZE = len(x_train_df)

print('ds size', DS_SIZE)

# create initial dataset from tensor slices using the dataset api

image_ds = tf.data.Dataset.from_tensor_slices(np.resize(x_train_df, (DS_SIZE, 110, 110, 1)))



# we will create an isolated volcano dataset to feed in 5 more times, with augmentation should distribute the labels nicely

volc_ind = y_train_df[y_train_df['Volcano?'] == 1].index

volca_image_ds = tf.data.Dataset.from_tensor_slices(np.resize(x_train_df.iloc[volc_ind], (len(volc_ind), 110, 110, 1)))

volca_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train_df.iloc[volc_ind]['Volcano?'].values, tf.int32))



repeat_amt = 5

# this doesn't duplicate data, but tells tensorflow how many times to repeat the data in the stream

image_ds = image_ds.concatenate(volca_image_ds.repeat(repeat_amt))

DS_SIZE = DS_SIZE + (repeat_amt * len(volc_ind))

print('new ds size', DS_SIZE)



# syncronized label dataset

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_train_df['Volcano?'].values, tf.int32))

label_ds = label_ds.concatenate(volca_label_ds.repeat(repeat_amt))



# preprocess all images (original + 4*volcano)

image_ds = image_ds.map(load_and_preprocess_image)
# zip images and labels then shuffle

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(DS_SIZE)



BATCH_SIZE = 64



# split into training and validation

train_size, val_size = int(0.99 * DS_SIZE), int(0.01 * DS_SIZE)



train_ds = (image_label_ds

    .take(train_size)

    .cache()

    .repeat()

    .batch(BATCH_SIZE)

    .map(augment_image)

    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))



val_ds = (image_label_ds

    .skip(train_size)

    .cache()

    .repeat()

    .batch(BATCH_SIZE)

    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
# model

def get_model():

    model = Sequential()

    

    model.add(Conv2D(64, (2, 2),  input_shape=IMAGE_SHAPE,  activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.3))

    

    model.add(Conv2D(96, (4, 4), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.45))



    model.add(Conv2D(128, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.4))

    

    model.add(Conv2D(128, (6, 6), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.4))

    

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(rate=0.35))

    model.add(Dense(1, activation="sigmoid"))

    

    return model



model = get_model()



# rmsprop seemed to have done better than other optimizers for me

model.compile(optimizer='rmsprop', 

              loss='binary_crossentropy',

              metrics=['acc'])
train_steps = (train_size//BATCH_SIZE)+1

val_steps = (val_size//BATCH_SIZE)+1



# fit

history = model.fit(train_ds,

                    steps_per_epoch=train_steps, 

                    epochs=50,

                    validation_data=val_ds, 

                    validation_steps=val_steps,

                    callbacks=None)
import matplotlib.pyplot as plt



plt.plot(history.history['acc'], label="train acc")

plt.plot(history.history['val_acc'], label="val acc")

plt.legend()

plt.show()
del x_train_df

del y_train_df



x_test_df = pd.read_csv('../input/volcanoes_test/test_images.csv', header=None)

y_test_df = pd.read_csv('../input/volcanoes_test/test_labels.csv')



TEST_DS_SIZE = len(x_test_df)



# run on testset

test_image_ds = tf.data.Dataset.from_tensor_slices(np.resize(x_test_df, (TEST_DS_SIZE, 110, 110, 1)))

test_image_ds = test_image_ds.map(load_and_preprocess_image)

test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y_test_df['Volcano?'].values, tf.int32))

test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds)).batch(BATCH_SIZE)



# got 98% accuracy on test set, good first try, will likely look at images and see where its missing the most

model.evaluate(test_image_label_ds, steps = (TEST_DS_SIZE//BATCH_SIZE)+1)
preds = model.predict(test_image_label_ds, steps = (TEST_DS_SIZE//BATCH_SIZE)+1)
import seaborn as sns

import matplotlib.pyplot as plt



# from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test_df['Volcano?'].values, np.round(preds))

conf_matrix_plt = print_confusion_matrix(conf_matrix, ["Not Volcano", "Volcano"], figsize = (5,4))