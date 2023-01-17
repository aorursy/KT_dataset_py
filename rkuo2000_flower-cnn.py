import os
import sys
from collections import Counter
import json
import cv2
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
# What is inside the labels.csv
DATA_PATH = "../input/flower-classification-dataset/"
df = pd.read_csv(os.path.join(DATA_PATH, 'labels.csv'))
df.tail()
X = []
y = []
for image, label in tqdm(zip(df.image_id.values, df.category.values), total=len(df)):
    try:
        xt = np.array(Image.open(os.path.join(DATA_PATH, f"files/{image}.jpg")).resize((128,128)))
        yt = label
        X.append(xt)
        y.append(yt)
    except:
        print(os.path.join(DATA_PATH, f"files/{image}.jpg"))
    
X = np.array(X)
y = np.array(y)

print(X.shape, y.shape)
files = os.listdir(os.path.join(DATA_PATH, 'files'))
print(f"The total number of files in the dataset are {len(files)}")
# Lets see that the number of images in the dataset equals to the provided labels
print(f"The total number of points in the labels.csv are {len(df)}")
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

print(train_y.shape, test_y.shape)
num_classes = 103
print(train_X.shape, train_y.shape)
# plot first few images
plt.figure(figsize=(12,12))
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(train_X[i])
# show the figure
plt.show()
plt.figure(figsize=(18,6))
df["category"].value_counts().plot(kind='bar')
!rm -rf preview
!mkdir preview
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img(os.path.join(DATA_PATH, f"files/0.jpg"))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="preview", save_prefix='f', save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
!ls preview
x=[]
for image in os.listdir('preview'):
    xt = np.array(Image.open(os.path.join("preview", image)).resize((128,128)))
    x.append(xt)    
    
# plot first few images
plt.figure(figsize=(12,12))
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(x[i])
# show the figure
plt.show()
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(
        train_X,
        train_y,
        batch_size=batch_size,
        shuffle=True
        )  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(
        test_X,
        test_y,
        shuffle=False,
        )
history = model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=validation_generator)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()