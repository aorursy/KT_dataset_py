import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization

from keras import backend as K

import os

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from random import randrange
enc = OneHotEncoder()

enc.fit([[0], [1]]) 

def names(number):

    if(number == 0):

        return 'Normal'

    else:

        return 'Malaria Infected'
os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images")
data = []

paths = []

ans = []

for r, d, f in os.walk(r"../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized"):

    for file in f:

        if '.png' in file:

            paths.append(os.path.join(r, file))



for path in paths:

    img = Image.open(path)

    x = img.resize((64,64))

    data.append(np.array(x))

    ans.append(enc.transform([[1]]).toarray())
paths = []

for r, d, f in os.walk(r"../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected"):

    for file in f:

        if '.png' in file:

            paths.append(os.path.join(r, file))



for path in paths:

    img = Image.open(path)

    x = img.resize((64,64))

    data.append(np.array(x))

    ans.append(enc.transform([[0]]).toarray())
data = np.array(data)

data.shape
ans = np.array(ans)

ans = ans.reshape(27558,2)
#splitting data into train and test sets. 3/4 train, 1/4 test.

x_train,x_test,y_train,y_test = train_test_split(data, ans, test_size=0.2, shuffle=True, random_state=69)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(64, 64, 3)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())
history = model.fit(x_train, y_train, epochs=20, batch_size=1000, verbose=1,validation_data=(x_test, y_test))
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Validation', 'Test'], loc='upper right')

plt.show()
img = Image.open(r"../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/C39P4thinF_original_IMG_20150622_105335_cell_16.png")

x = np.array(img.resize((64,64)))

x = x.reshape(1,64,64,3)

answ = model.predict_on_batch(x)

classification = np.where(answ == np.amax(answ))[1][0]

imshow(img)

print(str(answ[0][classification]*100) + '% Confidence This Is A ' + names(classification) + " Cell")
img = Image.open(r"../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/C148P109ThinF_IMG_20151115_112538_cell_205.png")

x = np.array(img.resize((64,64)))

x = x.reshape(1,64,64,3)

answ = model.predict_on_batch(x)

classification = np.where(answ == np.amax(answ))[1][0]

imshow(img)

print(str(answ[0][classification]*100) + '% Confidence This Is A ' + names(classification) + " Cell")