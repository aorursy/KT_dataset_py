# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
full_train_set = pd.read_csv("../input/train.csv")
final_set = pd.read_csv("../input/test.csv")

print(full_train_set.shape)

full_labels = full_train_set["label"]
full_pixels = full_train_set.drop(labels = ["label"], axis = 1)

train_labels, test_labels, train_pixels, test_pixels = train_test_split(full_labels, full_pixels)

print("Train Labels :", train_labels.shape)
print("Train Pixels :", train_pixels.shape)

print("Test Labels :", test_labels.shape)
print("Test Pixels :", test_pixels.shape)

print("Final Pixels :", final_set.shape)
train_pixels = train_pixels / 255.0
test_pixels = test_pixels / 255.0
final_set = final_set / 255.0

print(train_pixels.shape)
print(test_pixels.shape)
print(final_set.shape)
train_labels = keras.utils.to_categorical(train_labels, num_classes = 10)
train_pixels = train_pixels.values.reshape(-1, 28, 28, 1)

test_labels = keras.utils.to_categorical(test_labels, num_classes = 10)
test_pixels = test_pixels.values.reshape(-1, 28, 28, 1)

final_set = final_set.values.reshape(-1, 28, 28, 1)

print("Train Pixels: ", train_pixels.shape)
print("Test Pixels: ", test_pixels.shape)
print("Final Pixels: ", final_set.shape)
plt.imshow(train_pixels[50][:,:,0])
model = keras.Sequential()

#réseau de convolution
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

#layer d'applatissement de l'image en 1 dimension
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256))
model.add(keras.layers.Dropout(0.5))

#layer d'output
model.add(keras.layers.Dense(10, activation='softmax'))

#compilation du modèle avec une descente de gradient et la méthode des moindres carrés
model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(train_pixels, train_labels, epochs=20, batch_size=64)
loss, acc = model.evaluate(test_pixels, test_labels, batch_size=64)

print('Loss: ', loss)
print('Accuracy: ', acc)
limit = 25
width = 5
height = int(limit / width)

fig, axis = plt.subplots(height, width, sharex=True, sharey=True, figsize=(width * 5, height * 5))

final_labels = model.predict(final_set[:limit])

for i in range(limit):
    row = int(i / width)
    col = int(i % width)
    axis[row,col].imshow(final_set[i][:,:,0])
    axis[row,col].set_title("Label :{}\n".format(np.argmax(final_labels[i])))
    axis[row,col].title.set_fontsize(20)

plt.show()

#plt.imshow(test_set[0][:,:,0])
#print(np.argmax(test_labels[:10,0]))
final_labels = np.argmax(model.predict(final_set), axis=1).reshape((-1, 1))
indexes = [[i] for i in range(1, len(final_labels) + 1)]
data = np.concatenate((indexes, final_labels), axis=1)

results = pd.DataFrame(data=data, columns=['ImageId','Label'])

results.to_csv('submission.csv', index=False)
print(os.listdir("."))