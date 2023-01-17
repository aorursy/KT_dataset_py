# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Dataset path

DataDir = "../input/"



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import os

import cv2
train_dir = "train/train"

path = os.path.join(DataDir, train_dir)

for p in os.listdir(path):

    category = p.split(".")[0]

    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

    plt.imshow(img_array,cmap="gray")

    print(category, img_array.shape)

    break
# Dog = 0, Cat = 1

IMG_SIZ = 64



training_data = []



for p in os.listdir(path):

    category = p.split(".")[0]

    try:

        img_array = cv2.imread(os.path.join(path, p),cv2.IMREAD_GRAYSCALE)

        new_array = cv2.resize(img_array, (IMG_SIZ, IMG_SIZ))

        if category == 'dog':

            training_data.append([new_array, 0])

        else:

            training_data.append([new_array, 1])

    except Exception as e:

        pass
print(training_data[0][0], training_data[0][1])

plt.imshow(training_data[0][0], cmap="gray")
print(len(training_data))
X = []

y = []



for features, label in training_data:

    X.append(features)

    y.append(label)



X = np.array(X).reshape(-1, IMG_SIZ, IMG_SIZ, 1)

X = X / 255.0



print(X.shape[1:])
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = X.shape[1:]))

model.add(Dropout(rate = 0.2))

model.add(MaxPooling2D(2, 2))



model.add(Conv2D(128, (3, 3), activation = 'relu'))

model.add(Dropout(rate = 0.2))

model.add(MaxPooling2D(2, 2))



model.add(Conv2D(256, (3, 3), activation = 'relu'))

model.add(Dropout(rate = 0.2))

model.add(MaxPooling2D(2, 2))



model.add(Flatten())

model.add(Dense(256))



model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss="binary_crossentropy",

             optimizer="adam",

             metrics=['accuracy'])



model.fit(X, y, batch_size = 32, epochs = 30, validation_split = 0.1)
test_dir = "test1/test1"

path = os.path.join(DataDir, test_dir)



test_data = []

id = []



for p in os.listdir(path):

    try:

        id.append((p.split("."))[0])

        img_array = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE)

        new_array = cv2.resize(img_array, (IMG_SIZ, IMG_SIZ))

        test_data.append(new_array)

    except Exception as e:

        pass
for i in range(0, 5):

    print(id[i])

    plt.imshow(test_data[i], cmap="gray")

    plt.show()
test_data = np.array(test_data).reshape(-1, IMG_SIZ, IMG_SIZ, 1)

test_data = test_data / 255.0
y_pred = model.predict(test_data)

y_pred = np.round(y_pred, decimals = 2)

pred_labels = [1 if value > 0.5 else 0 for value in y_pred]
print(pred_labels)
df_submission=pd.DataFrame({"id":id})

df_submission["label"]=pred_labels

df_submission.info()
for i in range(0, len(id)):

    print(id[i], pred_labels[i])
df_submission.to_csv("cnn_submission.csv",index=False)