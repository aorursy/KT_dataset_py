import os

from PIL import Image

import numpy as np



DATA_DIR = "/kaggle/input/leapgestrecog/leapGestRecog"

IMAGE_SIZE_X = 160

IMAGE_SIZE_Y = 60



X = []

Y = []



for root,dirs, files in os.walk(DATA_DIR, topdown=False):

    print(".", end="")

    for name in files:

        path = os.path.join(root, name)

        if path.endswith("png"):

            

            # Resize & change into grey scale

            img = Image.open(path).convert('L')

            img = img.resize((IMAGE_SIZE_X, IMAGE_SIZE_Y))

            X.append(np.array(img))

            

            # Get gesture type for label

            category = name.split("_")[2][1]

            Y.append(int(category))



X = np.array(X, dtype = 'float32')

Y = np.array(Y)

print("done")
# One-hot encoding

from keras.utils import to_categorical

Y = to_categorical(Y)



# Normalization

X /= 255



# Split dataset for train/validation/test

from sklearn.model_selection import train_test_split

x_train, x_further, y_train, y_further = train_test_split(X, Y, test_size = 0.2)

x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size = 0.5)
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Flatten())

model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)

print("Accuracy: " + str(acc * 100) + "%")