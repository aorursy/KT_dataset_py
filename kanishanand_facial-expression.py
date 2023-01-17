import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import cv2

import shutil

import PIL

import seaborn as sns

import tensorflow as tf

from cv2 import CascadeClassifier

from keras import utils as np_utils

from sklearn.metrics import confusion_matrix,classification_report

from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image

from keras.optimizers import Adam,SGD

# import split_folders

from keras.models import Sequential,load_model

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from sklearn.model_selection import train_test_split

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ReduceLROnPlateau
path = "../input/ferg-db-datatset/FERG_DB_256/"

print(os.listdir(path))
print(os.listdir(path+"train/"))
labels = ['anger','disgust','fear','joy','neutral','sad','surprise']
img_height = 48

img_width = 48

img_channels = 1

batch_size = 32

epochs = 35
TRAIN_DIR = path + "train/"

print(TRAIN_DIR)

train_datagen = ImageDataGenerator(

    rescale = 1./255,

    zoom_range = 0.2,

    shear_range = 0.1,

    fill_mode = 'reflect',

    width_shift_range = 0.1,

    height_shift_range = 0.1

)



train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(img_height, img_width),

    color_mode="grayscale",

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=True,

    seed=42

)
sum = 0

for val in labels:

    count = len(os.listdir(TRAIN_DIR + val))

    sum = sum + count

    print(val,count)

    plt.bar(val,count)

    plt.xlabel('Emotions')

    plt.ylabel('Number of Images')

    plt.title('Distribution of images of training dataset')

train_size = sum    
VAL_DIR = path + "val/"

print(VAL_DIR)

val_datagen = ImageDataGenerator(

    rescale = 1./255,

)



val_generator = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=(img_height, img_width),

    color_mode="grayscale",

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=False

)
# TEST_DIR = "../input/test-imgs/test/"

# print(TEST_DIR)

# test_datagen = ImageDataGenerator(

#     rescale = 1./255,

# )



# test_generator = test_datagen.flow_from_directory(

#     TEST_DIR,

#     target_size=(128, 128),

#     color_mode="rgb",

#     batch_size=batch_size,

#     class_mode="categorical",

#     shuffle=False

# )
sum = 0

for val in labels:

    count = len(os.listdir(VAL_DIR + val))

    sum = sum + count

    print(val,count)

    plt.bar(val,count)

    plt.xlabel('Emotions')

    plt.ylabel('Number of Images')

    plt.title('Distribution of images of validation dataset')    

val_size = sum    
labels = []

for val in val_generator.class_indices:

    labels.append(val)
model = Sequential()

input_shape = (img_height,img_width,img_channels)



model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape,activation='relu', padding='same'))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size=(3,3),activation='relu',padding='same'))

model.add(Conv2D(128, kernel_size=(3,3),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



# model.add(Conv2D(filters=256, kernel_size=(3,3), activation="relu",padding="same"))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation="relu",padding="same"))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation="relu",padding="same"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



# model.add(Conv2D(filters=512, kernel_size=(3,3), activation="relu",padding="same"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), activation="relu",padding="same"))

# model.add(Conv2D(filters=512, kernel_size=(3,3), activation="relu",padding="same"))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))



model.add(Flatten())

model.add(Dense(units=512,activation="relu"))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(units=7, activation="softmax"))
# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer= 'Adam')
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',

                                            patience=2,

                                            factor=0.1,

                                            min_lr = 0.0000001,

                                            verbose = 1)
path_model='model_filter.h5' 

# fit the model

hist = model.fit_generator(

    train_generator,

    epochs = epochs,

    validation_data = val_generator,

    steps_per_epoch = train_size//batch_size,

    validation_steps = val_size//batch_size,

    callbacks = [

        ModelCheckpoint(filepath=path_model),

        learning_rate_reduction,

    ]

)
model.save('model.h5')
# model2 = load_model("../input/final-modelh5/model.h5")
# model = load_model("../input/model-th5/mode_t.h5")
# ### import matplotlib.pyplot as plt

# plt.plot(hist.history['accuracy'])

# plt.plot(hist.history['val_accuracy'])

# plt.title('model loss')

# plt.ylabel('loss')

# plt.xlabel('epoch')

# plt.legend(['train', 'test'], loc='upper left')

# plt.show()
score = model.evaluate_generator(train_generator)

print(score[1]*100)
score = model.evaluate_generator(val_generator)

print(score[1]*100)
# score = model2.evaluate_generator(test_generator)

# print(score[1]*100)
val_y = val_generator.classes
val_pred = model.predict_generator(val_generator)
y = []

for ind in range(val_pred.shape[0]):

    val = np.argmax(val_pred[ind])

    y.append(val)

y = np.array(y)
cfm = confusion_matrix(val_y,y)

print(cfm)

f = plt.figure(figsize=(13,13))

ax = f.add_subplot(111)

sns.heatmap(cfm, annot=True, ax = ax);

# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels');

ax.set_title('Confusion Matrix');

ax.xaxis.set_ticklabels(labels)

ax.yaxis.set_ticklabels(labels)
# classifier = CascadeClassifier('../input/haarcascade-frontalface-defaultxml/haarcascade_frontalface_default.xml')

classifier = CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
p = "../input/test-imgs/test/joy/images.jpeg"

pixels = cv2.imread(p)

bboxes = classifier.detectMultiScale(pixels)

for box in bboxes:

    x, y, width, height = box

    x2, y2 = x + width, y + height

    cv2.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)

img = pixels[y:y2,x:x2]

plt.imshow(img)    

cv2.imwrite('image.jpeg',img)
p = "image.jpeg"

img = image.load_img(p, target_size=(img_width, img_height),color_mode = 'grayscale')

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)
val = model.predict_classes(img)

labels[val[0]]
# p = "image.jpeg"

# img = image.load_img(p, target_size=(128, 128))

# img = image.img_to_array(img)

# img = np.expand_dims(img, axis=0)
# val = model2.predict_classes(img)

# labels[val[0]]
# shutil.rmtree("./split")