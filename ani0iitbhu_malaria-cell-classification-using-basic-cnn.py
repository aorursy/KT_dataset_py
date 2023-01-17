import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten
infected = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Parasitized')
uninfected = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Uninfected')
image_data = []
for img in infected:
    try:
        image = cv2.imread(os.path.join('../input/cell-images-for-detecting-malaria/cell_images/Parasitized',img),cv2.IMREAD_GRAYSCALE)
        image_resize = resize(image, (40,40), anti_aliasing=True)
        image_data.append([image_resize, 1])
    except AttributeError:
        print('')

for img in uninfected:
    try:
        image = cv2.imread(os.path.join('../input/cell-images-for-detecting-malaria/cell_images/Uninfected',img),cv2.IMREAD_GRAYSCALE)
        image_resize = resize(image, (40,40), anti_aliasing=True)
        image_data.append([image_resize, 0])
    except AttributeError:
        print('')
import random

random.shuffle(image_data)
X = []
y = []
for data, labels in image_data:
    X.append(data)
    y.append(labels)
X = np.array(X).reshape(-1, 40, 40, 1)
y = np.array(y).reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
model = Sequential()

model.add(Conv2D(128,(3,3),input_shape=X.shape[1:],activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, validation_split=0.1, epochs=8)
model.evaluate(X_test, y_test)
