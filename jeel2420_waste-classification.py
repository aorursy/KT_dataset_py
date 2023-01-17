# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os





# Any results you write to the current directory are saved as output.
train_data_o = "/kaggle/input/waste-classification-data/DATASET/TRAIN/O/"

train_data_r = "/kaggle/input/waste-classification-data/DATASET/TRAIN/R/"

test_data_o = "/kaggle/input/waste-classification-data/DATASET/TEST/O/"

test_data_r = "/kaggle/input/waste-classification-data/DATASET/TEST/R/"
from PIL import Image

import cv2
def generate_image(path,category):

    data = []

    label = []

    for filename in os.listdir(path):

        image = cv2.imread(path+filename)

        img_array = Image.fromarray(image, 'RGB')

        resize_img = img_array.resize((64,64))

        rotated45 = resize_img.rotate(45)

        rotated75 = resize_img.rotate(75)

        data.append(np.array(resize_img))

        data.append(np.array(rotated45))

        data.append(np.array(rotated75))

        if category == "O":

            label.append(0)

            label.append(0)

            label.append(0)

        else:

            label.append(1)

            label.append(1)

            label.append(1)

    return data, label
X_train_o, y_train_o = generate_image(train_data_o, "O") 

X_train_r, y_train_r = generate_image(train_data_r, "R") 

X_train = X_train_o + X_train_r

y_train = y_train_o + y_train_r

X_test_o, y_test_o = generate_image(test_data_o, "O") 

X_test_r, y_test_r = generate_image(test_data_r, "R") 

X_test = X_test_o + X_test_r

y_test = y_test_o + y_test_r

print((len(X_train),len(y_train)), (len(X_test), len(y_test)))
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
n = np.arange(X_train.shape[0])

np.random.shuffle(n)

X_train = X_train[n]

y_train = y_train[n]



n = np.arange(X_test.shape[0])

np.random.shuffle(n)

X_test = X_test[n]

y_test = y_test[n]



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train = X_train.astype(np.float32)

y_train = y_train.astype(np.float32)

X_train /= 255.0

X_test = X_test.astype(np.float32)

y_test = y_test.astype(np.float32)

X_test /= 255.0
import keras

from keras.models import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D

from keras.layers import Dropout, Dense, Flatten
model = Sequential()

model.add(Conv2D(64, (7,7), kernel_initializer='he_normal' , activation = "relu"))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(32, (5,5), kernel_initializer='he_normal' , activation = "relu"))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(32, (3,3), kernel_initializer='he_normal' , activation = "relu"))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(32, (3,3), kernel_initializer='he_normal' , activation = "relu"))

model.add(MaxPooling2D(2,2))



model.add(Dropout(0.01))

model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dense(256, activation = "relu"))

model.add(Dense(128, activation = "relu"))



model.add(Dropout(0.01))

model.add(Dense(2, activation = "sigmoid"))
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)



model.compile(loss='sparse_categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_test, y_test))
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()
score = model.evaluate(X_test,y_test,verbose=0)

print('Test Loss :',score[0])

print('Test Accuracy :',score[1])
predicted_classes = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix

import itertools



confusion_mtx = confusion_matrix(y_test, predicted_classes) 



plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('confusion_matrix')

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ['R','O'], rotation=90)

plt.yticks(tick_marks, ['R','O'])

#Following is to mention the predicated numbers in the plot and highligh the numbers the most predicted number for particular label

thresh = confusion_mtx.max() / 2.

for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):

    plt.text(j, i, confusion_mtx[i, j],

    horizontalalignment="center",

    color="white" if confusion_mtx[i, j] > thresh else "black")



plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')