# Basics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random



# Images

import os

from PIL import Image



# Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# Tensorflow

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from keras.utils import to_categorical
def get_pixel_data(filepath):

    """

    Get the pixel data from an image as a pandas DataFrame.

    """

    # Import libraries

    # import pandas as pd

    # from PIL import Image

    

    # Open the file

    image = Image.open(filepath)

    

    # Get the data and average between channels

    pixel_data = np.array(image.getdata())

    pixel_data = pixel_data.mean(axis = 1)

    pixel_data = pixel_data.reshape(1,32*32)

    pixel_data = pd.DataFrame(pixel_data, columns = np.arange(32*32))

    

    # Close the file

    image.close()

    

    return pixel_data
# Train data

path = "../input/train/train/"

train = pd.DataFrame()

for file in sorted(os.listdir(path)):

    image = get_pixel_data(path + file)

    train = train.append(image, ignore_index = True)



labels_train = pd.read_csv("../input/train.csv").sort_values("id")



# Test data

path = "../input/test/test/"

test = pd.DataFrame()

test_id = []

for file in sorted(os.listdir(path)):

    image = get_pixel_data(path + file)

    test  = test.append(image, ignore_index = True)

    test_id.append(file)
print("TRAIN---------------------")

print("Shape: {}".format(train.shape))

print("Label 0 (False): {}".format(np.sum(labels_train.has_cactus == 0)))

print("Label 1 (True):  {}".format(np.sum(labels_train.has_cactus == 1)))

print("TEST----------------------")

print("Shape: {}".format(test.shape))
# Create train set and validation set

random.seed(0)



idx = random.choices(range(17500), k = 10000)

X_train = train.iloc[idx] / 255           # Normalize

X_test  = train.drop(idx, axis = 0) / 255 # Normalize

test    = test / 255                      # Normalize

y_train = labels_train.iloc[idx,1]

y_test  = labels_train.drop(idx, axis = 0).iloc[:,1]

model = LogisticRegression(solver = "lbfgs", random_state = 0)

model.fit(X_train, y_train)

model.score(X_test, y_test)
model = RandomForestClassifier(n_estimators = 100, criterion = "entropy", random_state = 0)

model.fit(X_train, y_train)

model.score(X_test, y_test)
model = Sequential()

model.add(Dense(5, activation = "relu", input_shape = (1024,)))

model.add(Dense(10, activation = "relu"))

model.add(Dense(2,  activation = "sigmoid"))

model.summary()



model.compile(optimizer = "adam",

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])

model.fit(X_train, to_categorical(y_train), epochs = 5)

model.evaluate(X_test, to_categorical(y_test))[1]
X_train_cnn = np.array(X_train).reshape((X_train.shape[0], 32, 32, 1))

X_test_cnn  = np.array(X_test).reshape((X_test.shape[0], 32, 32, 1))

test_cnn    = np.array(test).reshape((test.shape[0], 32, 32, 1))



model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = 3, activation = "relu", input_shape = (32, 32, 1)))

model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(filters = 16, kernel_size = 3, activation = "relu"))

model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())

model.add(Dense(2,  activation = "softmax"))

model.summary()



model.compile(optimizer = "adam",

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])

model.fit(X_train_cnn, to_categorical(y_train), epochs = 10)

model.evaluate(X_test_cnn, to_categorical(y_test))[1]
# Make the predictions

preds = model.predict_classes(test_cnn)

print("Label 0 (False): {}".format(np.sum(preds == 0)))

print("Label 1 (True):  {}".format(np.sum(preds == 1)))



# Save the results

results = pd.DataFrame({"id" : test_id, "has_cactus": preds})

results.to_csv("submission.csv", index = False)