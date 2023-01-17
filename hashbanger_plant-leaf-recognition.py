from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import mobilenet_v2, inception_v3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pylab as plt
from keras import backend as K
from keras.models import load_model
import cv2
import numpy as np

data_dir = '../input/plantleaf/PlantVillage'
train_dir = '../input/plantleaf/PlantVillage'
validation_dir = '../input/plantleaf/PlantVillage'


def count(dir, counter=0):
    "returns number of files in dir and subdirs"
    for pack in os.walk(dir):
        for f in pack[2]:
            counter += 1
    return dir + " : " + str(counter) + "files"


print('total images for training :', count(train_dir))
print('total images for validation :', count(validation_dir))


IMAGE_SIZE = (256, 256)
BATCH_SIZE = 10
classes = ["Pepper__bell___Bacterial_spot",
           "Pepper__bell___healthy",
           "Potato___Early_blight",
           "Potato___healthy",\
           "Potato___Late_blight",
           "Tomato__Target_Spot",
           "Tomato__Tomato_mosaic_virus",
           "Tomato__Tomato_YellowLeaf__Curl_Virus",
           "Tomato_Bacterial_spot",\
           "Tomato_Early_blight",
           "Tomato_healthy",
           "Tomato_Late_blight",
           "Tomato_Leaf_Mold",
           "Tomato_Septoria_leaf_spot",
           "Tomato_Spider_mites_Two_spotted_spider_mite"]

print(classes)

print('Number of classes:', len(classes))

train_data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = test_data_gen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')
inputShape = (256,256,3)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (3,256,256)
    chanDim = 1
n_classes = 15
# RUN WHEN TRAINING 

# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))

# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation("relu"))

# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(n_classes))
# model.add(Activation("softmax"))

# model.save('kaggle_plant_model.h5')


# model.summary()
# RUN FOR TRAINING 

# opt = Adam(lr=0.001)

# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# EPOCHS = 100


# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     verbose=1)

# RUN FOR LOADING 

# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))

# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation("relu"))

# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(n_classes))
# model.add(Activation("softmax"))

#opt = Adam(lr=0.001)

#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# RUN FOR LOADING
# model.load_model(<path>)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.show()
# def prepare(img_path):
#     img = cv2.imread(img_path)
#     x = np.asarray(img)
#     x = 1./255
#     return np.expand_dims(x, axis=0)
img = cv2.imread('../input/plantleaf/PlantVillage/Tomato_Leaf_Mold/50e1906e-24da-493d-88b0-3110e778a26a___Crnl_L.Mold 7135.JPG')
plt.imshow(img)
img = img.reshape(1,256,256,3)
print(classes[model.predict_classes(img)[0]])
# result = model.predict_classes(
#     [prepare('/content/drive/My Drive/LEAF_PROJECT/test_images/1.jpg')])
# disease = cv2.imread('/content/drive/My Drive/LEAF_PROJECT/test_images/1.jpg')
# plt.imshow(disease)
# print(classes[int(result)])