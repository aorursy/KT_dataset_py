import numpy as np
import pandas as pd
import tensorflow

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold, train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Used to track how long the model is training
import time
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Read in the different datafiles
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
# Set 'label' column as targets of trainset
Y_train = train["label"]

# Drop 'label' column from trainset, to only leave the features (aka the pixels)
X_train = train.drop(labels = ["label"],axis = 1)
# Print the distribution of the digits present in the trainset
print('Label   Count    Percentage')
for i in range(0,10):
    print("%d       %d     %.2f" % (i, Y_train.value_counts()[i], round(Y_train.value_counts(normalize=True)[i]*100, 2)))
# Divide values by 255 to get an input value between 0 and 1 for every pixel
X_train = X_train / 255.0
test = test / 255.0
# Creating copies to use later in the kFold approach
X_train_K = X_train.copy()
Y_train_K = Y_train.copy()
test_K = test.copy()
# reshaping the data to rank 4 so I can use Data Augmentation

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

X_train.shape
Y_train = keras.utils.to_categorical(Y_train, num_classes=10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=42)
print(X_train.shape)
print(Y_train.shape)
model_A = keras.models.Sequential()

# Flattening the data so I can fit the rank 4 data
model_A.add(Flatten())

# For some reason got the best score using only one layer
# Kept making the layer denser and it kept improving my score, it stopt improving arround 1024
model_A.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add dropout to my layer to make it less vulnerable to overfitting
model_A.add(Dropout(0.5))

model_A.add(keras.layers.Dense(10, activation="softmax"))
model_A.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Add the parameters for Data Augmentation
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

# Fit the Data Augmentation to my training set
datagen.fit(X_train)
X_train.shape
history = model_A.fit_generator(datagen.flow(X_train,Y_train, batch_size=32),
                              epochs = 55, validation_data = (X_val,Y_val))
predictions_A = model_A.predict_classes(test)
print(predictions_A)
my_submission_A = pd.DataFrame({'ImageId': list(range(1,len(predictions_A)+1)), 'label': predictions_A})

# you could use any filename. We choose submission here
my_submission_A.to_csv('submission_A.csv', index=False)
# Remove anything but the values of the X_train dataframe
X_train_K = X_train_K.values

print(X_train_K.shape)
print(Y_train_K.shape)
kf = KFold(n_splits = 10,
           shuffle=True)
# Create a copy of the labels to use for printing samples of the train and testsets
# Samples are printed to give a general idea of train and testsets of the different folds
Y_labels = Y_train_K
for train_idx, test_idx in kf.split(X_train_K):
    _train = plt.figure(figsize=(20,2))
    for i in range(1,11):
        ax = _train.add_subplot(1, 10, i)
        ax.imshow(X_train_K[train_idx[i-1]].reshape(28, 28))
        ax.set_xlabel(Y_labels[train_idx[i-1]])
    _train.suptitle('Trainsample of in total %d records' % len(train_idx), fontsize=14)
    plt.show()
    
    _test = plt.figure(figsize=(20,2))
    for i in range(1,11):
        ax = _test.add_subplot(1, 10, i)
        ax.imshow(X_train_K[test_idx[i-1]].reshape(28, 28))
        ax.set_xlabel(Y_labels[test_idx[i-1]])
    _test.suptitle('Testsample of in total %d records' % len(test_idx), fontsize=14)
    plt.show()
# Convert Y train values into a matrix with 10 columns, a column for each class
#   (Comparable to hot-encoding)
Y_train_K = keras.utils.to_categorical(Y_train_K, num_classes=10)
model_K = keras.models.Sequential()

model_K.add(keras.layers.Dense(784, activation='relu', input_shape=(784,)))
model_K.add(keras.layers.Dense(800, activation="relu"))
model_K.add(keras.layers.Dense(10, activation="softmax"))
# Configure the learning process
model_K.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy', 'mse']
)
# Summarize the model, this gives information about the amount of parameters (weights & biases)
model_K.summary()
# Keep track of the running time by storing the starttime
start_time = time.time()

# Fit the model for every fold in the kFold
for train_idx, test_idx in kf.split(X_train_K):
    model_K.fit(
        X_train_K[train_idx],
        Y_train_K[train_idx],
        batch_size=32,
        epochs=15,
        validation_data=(X_train_K[test_idx], Y_train_K[test_idx])
    )
    
# Calculate the runtime by substracting the starttime from the current time
runtime = time.time() - start_time
print("/n--- Runtime of %s seconds ---" % (runtime))
# Use the trained neural network to identify the digits in the testset
predictions_K = model_K.predict_classes(test.values)
print(predictions_K)
# Create a dataframe from the predictions, made by the neural network
my_submission_K = pd.DataFrame({'ImageId': list(range(1,len(predictions_K)+1)), 'label': predictions_K})

# Save the predictions in the file 'submission.csv'
my_submission_K.to_csv('submission_K.csv', index=False)
