import pandas as pd

import numpy as np

import pathlib



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

plt.style.use('seaborn')



from scipy import ndimage

import cv2

import os



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.models import Sequential, load_model

from keras.applications import VGG19, VGG16, ResNet50



from sklearn.metrics import accuracy_score



import random
ac = pd.read_csv("../input/doom-crossing/animal_crossing_dataset.csv")

doom = pd.read_csv("../input/doom-crossing/doom_crossing_dataset.csv")



ac_filePath = "../input/doom-crossing/animal_crossing/"

doom_filePath = "../input/doom-crossing/doom/"
ac.head()
doom.head()
ac_fileNames = list(ac.filename.values)

doom_fileNames = list(doom.filename.values)
random.seed(123)

ac_subset = random.sample(ac_fileNames, 25)

doom_subset = random.sample(doom_fileNames, 25)



def plot_images(file_subset, ac_flag):

    plt.figure(figsize=(15,15))

    for i in range(25):

        if ac_flag:

            load_img = mpimg.imread(os.path.join(ac_filePath,file_subset[i]))

        else:             

            load_img = mpimg.imread(os.path.join(doom_filePath,file_subset[i]))

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(load_img)

    plt.show()
plot_images(ac_subset, True)
plot_images(doom_subset, False)
IMG_HEIGHT = 128

IMG_WIDTH = 128

COLOURS = 3



N_CLASS = 2

CLASS_NAMES = ["animal_crossing", "doom"]

BATCH_SIZE = 32
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,

                                                                horizontal_flip = True,

                                                                validation_split=0.2)





test_ds = img_generator.flow_from_directory(directory = "../input/ac-doom-testing/test2/test",

                                            shuffle = False,

                                            target_size=(IMG_HEIGHT, IMG_WIDTH),

                                            classes = CLASS_NAMES)



train_ds = img_generator.flow_from_directory(batch_size = BATCH_SIZE,

                                              directory = "../input/doom-crossing",

                                              shuffle=True,

                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                              classes= CLASS_NAMES,

                                              subset='training')



valid_ds = img_generator.flow_from_directory(batch_size = BATCH_SIZE,

                                              directory = "../input/doom-crossing",

                                              shuffle=True,

                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                              classes= CLASS_NAMES,

                                              subset='validation')
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()

    

sample_training_images, _ = next(train_ds)

plotImages(sample_training_images[:5])
def plot_fit(hist, metric):

    train_met = hist.history[metric]

    valid_met = hist.history['val_' + metric]

    

    plt.figure(figsize=(12,6))

    plt.plot(train_met)

    plt.plot(valid_met)

    plt.xlabel("Epoch Num")

    plt.legend(["train", "valid"])

    plt.show()
NUM_EPOCHS = 15

STEPS_PER_EPOCH = np.ceil(train_ds.samples // BATCH_SIZE)

VALID_STEPS = np.ceil(valid_ds.samples // BATCH_SIZE)
callbacks0 = [EarlyStopping(patience = 5),

             ReduceLROnPlateau(monitor = 'val_loss', patience = 5),

             ModelCheckpoint('../working/model.best.hdf5', save_best_only=True)]
model0 = Sequential([

    Conv2D(128, 3, padding='same',

                  activation='relu',

                  input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(128, 3, padding='same', activation='relu'),

    BatchNormalization(),

    Dropout(0.2),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu', dilation_rate = (2,2)),

    MaxPooling2D(),

    BatchNormalization(),

    Flatten(),

    Dense(128, activation = 'relu'),

    Dropout(0.2),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])



model0.summary()
model0.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



history0 = model0.fit(

    train_ds,

    steps_per_epoch= STEPS_PER_EPOCH,

    epochs= NUM_EPOCHS,

    validation_data = valid_ds,

    validation_steps = VALID_STEPS,

    callbacks = callbacks0

)
plot_fit(history0, 'accuracy')
plot_fit(history0, 'loss')
model0 = load_model('../working/model.best.hdf5')

predictions = model0.predict(test_ds)

prob_doom = [x[1] for x in predictions]

prob_ac = [x[0] for x in predictions]

predictions = [np.argmax(x) for x in predictions]
pred_df = pd.DataFrame({"file": test_ds.filenames,

                        "class": test_ds.classes,

                        "label": predictions,

                        "ac_prob": prob_ac,

                        "doom_prob": prob_doom})
pred_df
accuracy_score(pred_df.iloc[:,1], pred_df.iloc[:,2])
plt.figure(figsize = (15, 15))

for i in range(32):

    path = os.path.join("../input/ac-doom-testing/test2/test",pred_df.iloc[i,0])

    load_img = mpimg.imread(path)

    plt.subplot(8,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(load_img)

    prob = round(pred_df.iloc[i,3] if pred_df.iloc[i,2] == 0 else pred_df.iloc[i, 4], 5)

    if pred_df.iloc[i,2] == pred_df.iloc[i,1]: # correct classification

        plt.xlabel(CLASS_NAMES[pred_df.iloc[i,2]] + " prob: " + str(prob), color = 'blue')

    else:

        plt.xlabel(CLASS_NAMES[pred_df.iloc[i,2]] + " prob: " + str(prob), color = 'red')

plt.show()
callbacks1 = [EarlyStopping(patience = 4),

             ReduceLROnPlateau(monitor = 'val_loss', patience = 5),

             ModelCheckpoint('../working/restnet50model.best.hdf5', save_best_only=True)]



model1 = Sequential([

    ResNet50(include_top = False, pooling = 'avg', weights='imagenet'),

    Flatten(),

    Dense(256,activation=('relu')),

    Dropout(0.3),

    Dense(128,activation=('relu')),

    BatchNormalization(),

    Dense(2, activation = "softmax")

])



model1.summary()
model1.compile(optimizer=tf.keras.optimizers.Adam(),

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history1 = model1.fit(train_ds,

    steps_per_epoch= STEPS_PER_EPOCH,

    epochs= NUM_EPOCHS,

    validation_data = valid_ds,

    validation_steps = VALID_STEPS,

    callbacks = callbacks1

)
predictions1 = model1.predict(test_ds)

prob_doom1 = [x[1] for x in predictions1]

prob_ac1 = [x[0] for x in predictions1]

predictions1 = [np.argmax(x) for x in predictions1]
pred_df1 = pd.DataFrame({"file": test_ds.filenames,

                        "class": test_ds.classes,

                        "label": predictions1,

                        "ac_prob": prob_ac1,

                        "doom_prob": prob_doom1})
pred_df1
accuracy_score(pred_df1.iloc[:,1], pred_df1.iloc[:,2])
plt.figure(figsize = (15, 15))

for i in range(32):

    path = os.path.join("../input/ac-doom-testing/test2/test",pred_df1.iloc[i,0])

    load_img = mpimg.imread(path)

    plt.subplot(8,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(load_img)

    prob = round(pred_df1.iloc[i,3] if pred_df1.iloc[i,2] == 0 else pred_df1.iloc[i, 4], 5)

    if pred_df1.iloc[i,2] == pred_df1.iloc[i,1]:    

        plt.xlabel(CLASS_NAMES[pred_df1.iloc[i,2]] + " prob: " + str(prob), color = 'blue')

    else:

        plt.xlabel(CLASS_NAMES[pred_df1.iloc[i,2]] + " prob: " + str(prob), color = 'red')

plt.show()