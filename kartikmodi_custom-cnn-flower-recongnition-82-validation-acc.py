# Importing required libraries

import os 
import numpy as np
import pickle 
from tensorflow import keras 
import cv2
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator
os.getcwd()
data = []
img_size = 125
categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
def create_data():
    for category in categories:
        path = os.path.join('../input/flowers-recognition/flowers', category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img))
            try:
                new_arr = cv2.resize(img_arr, (img_size, img_size))
            except cv2.error as e:
                print('Not valid')
            cv2.waitKey()
            
            data.append([new_arr, class_num])
create_data()
random.shuffle(data)
X=[]
y=[]

for features, labels in data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y)
# Checking if everything is working 
print('Shape of X: ', X.shape)
print('Shape of y: ', y.shape)
pickle_out = open('X.pickle', 'wb')

pickle.dump(X, pickle_out)

pickle_out_2 = open('y.pickle', 'wb')

pickle.dump(y, pickle_out_2)
X = X / 255.0

plt.imshow(X[4])
y[4]

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
le = LabelEncoder()
y= le.fit_transform(y)
y = to_categorical(y,5)
y.shape

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, random_state= 7)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) 
datagen.fit(X_train)

model = keras.models.Sequential([keras.layers.Conv2D(32, 5, activation='relu', padding='SAME', input_shape=X.shape[1:]),
                                keras.layers.MaxPooling2D(2),
                                # keras.layers.Dropout(0.2),
                                keras.layers.Conv2D(64, 3, activation='relu', padding='SAME'),
                                keras.layers.MaxPooling2D(pool_size=2),
                                # keras.layers.Dropout(0.2),
                                
                                keras.layers.Conv2D(96, 3, activation="relu", padding="same"),
                                keras.layers.MaxPooling2D(pool_size=2),
                                # keras.layers.Dropout(0.2),
                                keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                                # keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                                keras.layers.MaxPooling2D(pool_size=2),
                                # keras.layers.Dropout(0.2),
                                keras.layers.Flatten(),
                                
                                keras.layers.Dense(500, activation='relu'),
                                keras.layers.Dropout(0.7),
                                keras.layers.Dense(5, activation='softmax')
                                ])
model.summary()
#Compiling the model

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128), epochs=70, validation_data=(X_valid, y_valid))
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
