# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visulation
import glob # For including images
import cv2 # OpenCV 
import tensorflow as tf # Machine learning lib
from tensorflow import keras # Tensorflow high-level api
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import training dataset
training_fruit_img = []
training_label = []
for dir_path in glob.glob("../input/*/fruits-360/Training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_fruit_img.append(img)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
len(np.unique(training_label))
# Import test dataset
test_fruit_img = []
test_label = []
for dir_path in glob.glob("../input/*/fruits-360/Test/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_fruit_img.append(img)
        test_label.append(img_label)
test_fruit_img = np.array(test_fruit_img)
test_label = np.array(test_label)
len(np.unique(test_label))
# Import multiple-test dataset
test_fruits_img = []
tests_label = []
for img_path in glob.glob(os.path.join("../input/*/fruits-360/test-multiple_fruits", "*.jpg")):
     img = cv2.imread(img_path)
     img = cv2.resize(img, (64, 64))
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     test_fruits_img.append(img)
     tests_label.append(img_label)
test_fruits_img = np.array(test_fruits_img)
tests_label = np.array(tests_label)
len(np.unique(tests_label))
label_to_id = {v : k for k, v in enumerate(np.unique(training_label))}
id_to_label = {v : k for k, v in label_to_id.items()}
training_label_id = np.array([label_to_id[i] for i in training_label])
test_label_id = np.array([label_to_id[i] for i in test_label])
test_label_id
training_fruit_img, test_fruit_img = training_fruit_img / 255.0, test_fruit_img / 255.0 
plt.imshow(training_fruit_img[0])
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), input_shape = (64, 64, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation = "relu"))
model.add(keras.layers.Dense(75, activation = "softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer = keras.optimizers.Adamax(), metrics = ['accuracy'])
tensorboard = keras.callbacks.TensorBoard(log_dir = "./Graph", histogram_freq = 0, write_graph = True, write_images = True)
model.fit(training_fruit_img, training_label_id, batch_size = 128, epochs = 10, callbacks = [tensorboard])
loss, accuracy = model.evaluate(test_fruit_img, test_label_id)
print("\n\nLoss:", loss)
print("Accuracy:", accuracy)
model.save("model.h5")
predictions = model.predict(test_fruits_img)
plt.figure(figsize = (30, 30))
for i in range(45):
    plt.subplot(9,5, i + 1)
    plt.xlabel("{}".format(id_to_label[np.argmax(predictions[i])]))
    plt.imshow(test_fruits_img[i])