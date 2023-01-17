# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import psutil

import matplotlib.pyplot as plt

from tqdm import tqdm

from math import ceil

from sklearn.model_selection import train_test_split

from tensorflow import set_random_seed

import tensorflow as tf

import keras

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Flatten, Dense

from keras.models import Model

from keras.layers import BatchNormalization

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import Input

from keras import backend as K



print(os.listdir('/kaggle/input'))

print(os.listdir('/kaggle/input/inceptionv3/'))

SEED = 7

np.random.seed(SEED)

set_random_seed(SEED)

dir_path = "/kaggle/input/"

IMG_DIM = 299  # 224

BATCH_SIZE = 8

CHANNEL_SIZE = 3

NUM_EPOCHS = 60

TRAIN_DIR = 'train_images'

TEST_DIR = 'test_images'

FREEZE_LAYERS = 2  # freeze the first this many layers for training

CLASSS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

NUM_CLASSS = 5
ROOT_PATH = '/kaggle/input/aptos2019-blindness-detection'

TRAIN_PATH = '/kaggle/input/aptos2019-blindness-detection/' + TRAIN_DIR 

TEST_PATH = '/kaggle/input/aptos2019-blindness-detection/' + TEST_DIR 

dir_path = ROOT_PATH + '/'
# print names of train images

train_img_names = glob.glob(TRAIN_PATH + '/*.png')

#print(train_img_names)



df_train = pd.read_csv(ROOT_PATH + '/train.csv')

#print(df_train)
# print names of test images

test_img_names = glob.glob(TEST_PATH + '/*.png')

#print(test_img_names)

df_test = pd.read_csv(ROOT_PATH + '/test.csv')

#print(df_test)
# Function to show one image



def draw_img(imgs, target_dir, class_label='0'):

    for row in enumerate(imgs.iterrows()):

        name = row[1][1]['id_code'] + '.png'

        print(name)

        plt.figure(figsize=(15,10))

        img = plt.imread(dir_path + target_dir + '/' + name)

        plt.imshow(img)

        plt.title(class_label)

        plt.show()

        del img

        gc.collect
# Showing the class 0 image randomly

CLASS_ID = 0

draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])
# Showing the class 1 image randomly

CLASS_ID = 1

draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])
# Showing the class 2 image randomly

CLASS_ID = 2

draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])
# Showing the class 3 image randomly

CLASS_ID = 3

draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])
# Showing the class 4 image randomly

CLASS_ID = 4

draw_img(df_train[df_train.diagnosis == CLASS_ID].sample(n=1), 'train_images', CLASSS[CLASS_ID])
gc.collect()
# Split Dataset



x_train, x_test, y_train, y_test = train_test_split(df_train.id_code, df_train.diagnosis, test_size=0.2,

                                                    random_state=SEED, stratify=df_train.diagnosis)
input_tensor = Input(shape = (299, 299, 3))



# create the base pre-trained model

base_model = InceptionV3(weights=None, include_top=False, input_tensor=input_tensor)

base_model.load_weights('/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')



# add a global spatial average pooling layer

x = base_model.output

output = BatchNormalization()(x)

x = GlobalAveragePooling2D()(x)



# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)



# and a logistic layer -- let's say we have 200 classes

predictions = Dense(NUM_CLASSS, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



for layer in model.layers:

    layer.trainable = True

    

print(model.summary())
epochs = 25

lrate = 0.01

decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("available RAM:", psutil.virtual_memory())

gc.collect()

print("available RAM:", psutil.virtual_memory())



df_train.id_code = df_train.id_code.apply(lambda x: x + ".png")

df_test.id_code = df_test.id_code.apply(lambda x: x + ".png")

df_train['diagnosis'] = df_train['diagnosis'].astype('str')
# Data Generator

train_datagen = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.15, horizontal_flip=True,

                                         vertical_flip=True, rotation_range=360, zoom_range=0.2, shear_range=0.1)
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,

                                                    directory= TRAIN_PATH + '/',

                                                    x_col='id_code',

                                                    y_col='diagnosis',

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='categorical',

                                                    target_size=(IMG_DIM, IMG_DIM),

                                                    subset='training',

                                                    shaffle=True,

                                                    seed=SEED

                                                    )

valid_generator = train_datagen.flow_from_dataframe(dataframe=df_train,

                                                    directory= TRAIN_PATH + '/',

                                                    x_col='id_code',

                                                    y_col='diagnosis',

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='categorical',

                                                    target_size=(IMG_DIM, IMG_DIM),

                                                    subset='validation',

                                                    shaffle=True,

                                                    seed=SEED

                                                    )

#del x_train

# # del x_test

#del y_train

# del y_test

gc.collect()

#  color_mode= "grayscale",
NUB_TRAIN_STEPS = train_generator.n // train_generator.batch_size

NUB_VALID_STEPS = valid_generator.n // valid_generator.batch_size



NUB_TRAIN_STEPS, NUB_VALID_STEPS
eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')

# Reducing the Learning Rate if result is not improving. 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',

                              verbose=1)
history = model.fit_generator(generator=train_generator,

                                     steps_per_epoch=NUB_TRAIN_STEPS,

                                     validation_data=valid_generator,

                                     validation_steps=NUB_VALID_STEPS,

                                     epochs=NUM_EPOCHS,

                                     #                            shuffle=True,  

                                     callbacks=[eraly_stop, reduce_lr],

                                     verbose=1)

gc.collect()
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

(eval_loss, eval_accuracy) = tqdm(

    model.evaluate_generator(generator=valid_generator, steps=NUB_VALID_STEPS, pickle_safe=False))

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))

print("[INFO] Loss: {}".format(eval_loss))
'''scores = model.evaluate(x_test, y_test, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))'''
accu = history.history['acc']

val_acc = history.history['val_acc']



plt.plot(accu, label="Accuracy")

plt.plot(val_acc)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Acc', 'val_acc'])

plt.plot(np.argmax(history.history["val_acc"]), np.max(history.history["val_acc"]), marker="x", color="r",

         label="best model")

plt.show()
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(history.history["loss"], label="loss")

plt.plot(history.history["val_loss"], label="val_loss")

plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",

         label="best model")

plt.xlabel("Epochs")

plt.ylabel("log_loss")

plt.legend();
test_datagen = image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True)



test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,

                                                  directory= TEST_PATH  + '/',

                                                  x_col="id_code",

                                                  target_size=(IMG_DIM, IMG_DIM),

                                                  batch_size=1,

                                                  shuffle=False,

                                                  class_mode=None,

                                                  seed=SEED)

# del df_test

print(df_test.shape[0])

# del train_datagen

# del traabsin_generator

gc.collect()
# evaluating the model on test data



tta_steps = 5

preds_tta = []

for i in tqdm(range(tta_steps)):

    test_generator.reset()

    preds = model.predict_generator(generator=test_generator, steps=ceil(df_test.shape[0]))

    #     print('Before ', preds.shape)

    preds_tta.append(preds)

#     print(i,  len(preds_tta))
final_pred = np.mean(preds_tta, axis=0)

predicted_class_indices = np.argmax(final_pred, axis=1)

len(predicted_class_indices)
results = pd.DataFrame({"id_code": test_generator.filenames, "diagnosis": predicted_class_indices})

results.id_code = results.id_code.apply(lambda x: x[:-4])  # results.head()

results.to_csv("submission.csv", index=False)
print(results)