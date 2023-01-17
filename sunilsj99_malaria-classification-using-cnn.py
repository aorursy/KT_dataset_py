# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np

import pandas as pd

import cv2

from PIL import Image

import seaborn as sns

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
path0 = '../input/cell_images/cell_images/Parasitized/'

path1 = '../input/cell_images/cell_images/Uninfected/'



para = os.listdir(path0)

unin = os.listdir(path1)
plt.imshow(cv2.imread(path0+para[0]))



img0 = cv2.imread(path0+para[0])

img0.shape
Images = []

Labels = []



for p in para:

    try:

        img = cv2.imread(path0+p)

        img_array = Image.fromarray(img, 'RGB')

        img_re = img_array.resize((60,60))

        Images.append(np.array(img_re))

        Labels.append(0)

    except AttributeError:

        print("")



for u in unin:

    try:

        img = cv2.imread(path1+u)

        img_array = Image.fromarray(img, 'RGB')

        img_re = img_array.resize((60,60))

        Images.append(np.array(img_re))

        Labels.append(1)

    except AttributeError:

        print("")
len(Images)
plt.imshow(Images[0])



Images[0].shape
Images = np.array(Images)

Labels = np.array(Labels)
Images.shape
Images = Images/255.
X_train, X_test , y_train , y_test = train_test_split(Images, Labels, test_size=0.2, random_state = 42)

X_train, X_val , y_train , y_val = train_test_split(X_train,y_train, test_size=0.2, random_state = 42)
print(str(len(X_train)) + " " + str(len(y_train)))

print(str(len(X_test)) + " " + str(len(y_test)))

print(str(len(X_val)) + " " + str(len(y_val)))
plt.figure(figsize=(15,5))

n=0

for i,j in zip([y_train,y_test, y_val] , ['train labels', 'test labels','validation labels']):

    n += 1

    plt.subplot(1, 3, n)

    sns.countplot(x=i)

    plt.title(j)

plt.show()
y_test[:10]
X_train = np.array(X_train)

X_test = np.array(X_test)

X_val = np.array(X_val)
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Activation

from keras import optimizers

from keras.models import Sequential
model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(60,60,3), activation='relu'))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))
model.summary()

  
model.compile(loss='binary_crossentropy',

            optimizer=optimizers.Adam(lr=0.0001),

            metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, verbose=1 ,epochs=15)
# loss

plt.figure(figsize=(15,7))

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.legend()

plt.show()
plt.figure(figsize=(15,7))

plt.plot(history.history['acc'], label='train acc')

plt.plot(history.history['val_acc'], label='val acc')

plt.legend()

plt.show()
#Predicting one image label

predictions = model.predict(X_test)
predictions
from sklearn import metrics
confusion_mat = metrics.confusion_matrix(y_test,predictions.round())

confusion_mat
df_cm = pd.DataFrame(confusion_mat, index = [i for i in "01"],  columns = [i for i in "01"])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True, cmap="YlGnBu")