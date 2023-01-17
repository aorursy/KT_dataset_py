import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, Dropout

import cv2

from sklearn.model_selection import train_test_split



print(os.listdir("../input/cell_images/cell_images/"))
data = []

uninfected = os.listdir("../input/cell_images/cell_images/Uninfected")

parasitized = os.listdir("../input/cell_images/cell_images/Parasitized")



for i in uninfected:

    data.append(["../input/cell_images/cell_images/Uninfected/"+i,0])

for i in parasitized:

    data.append(["../input/cell_images/cell_images/Parasitized/"+i,1])

random.shuffle(data)

image = [i[0] for i in data]

label = [i[1] for i in data]

del data
X_train, X_test, Y_train, Y_test = train_test_split(image, label, test_size=0.1, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=46)
def get_image(path):

    im = cv2.imread(path,1)

    im = cv2.resize(im,(60,60))

    im = im/255

    return im

X_images = []

Y_images = []

X_val_im = []

Y_val_im = []

c = 0

for i in range(len(X_train)):

    try:

        X_images.append(get_image(X_train[i]))

        Y_images.append(Y_train[i])

        c += 1

    except:

        print('c: ' + str(c))

Y_train = Y_images

c = 0

for i in range(len(X_val)):

    try:

        X_val_im.append(get_image(X_val[i]))

        Y_val_im.append(Y_val[i])

    except:

        print('c: ' + str(c))

Y_val = Y_val_im
X_images = np.array(X_images)

X_val_im = np.array(X_val_im)
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(60,60,3)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Conv2D(16, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_images, Y_train, validation_data=(X_val_im, Y_val), epochs=10)
del X_train, Y_train, X_val, Y_val, X_images, X_val_im
X_images = []

Y_images = []

for i in range(len(X_test)):

    X_images.append(get_image(X_test[i]))

    Y_images.append(Y_test[i])

    c += 1

X_images = np.array(X_images)
pred = np.rint(model.predict(X_images))
c =0

for i in range(len(pred)):

    if(pred[i] == Y_images[i]):

        c+=1

print(str(np.round((c*100)/len(pred),2))+'% of test cases were predicted successfully')