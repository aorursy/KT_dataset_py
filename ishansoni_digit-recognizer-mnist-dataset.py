# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import required libraries

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Basic ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Deep Learning libraries
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

%matplotlib inline
plt.style.use('ggplot')
sns.set_style('whitegrid')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

display(train.info())

display(test.info())

display(train.head(n = 2))
display(test.head(n = 2))
# Split the train dataset into features and labels

features_train = train.iloc[:, 1:]
labels_train = train.iloc[:, 0:1].values
# Normalize the data and reshape it

features_train = features_train / 255.0
test = test / 255.0

# features_train.iloc[1, :].values.shape -> 1D array with shape (784,)

sampleImageIndex = 1010

sampleImagePixelMap = features_train.iloc[sampleImageIndex, :].values.reshape(28, 28)
print(sampleImagePixelMap.shape)

print("The below Image should be a ", labels_train[sampleImageIndex])
g = plt.imshow(sampleImagePixelMap)
# Reshaping Contd.

features_train = features_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

print(features_train.shape)
print(test.shape)

print("The below Image should still be a ", labels_train[sampleImageIndex])
g = plt.imshow(features_train[sampleImageIndex, :, :, 0])
# Lets look at the label distribution in our training data set

sns.countplot(x = "label", data = train)
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.xlabel("Number")
plt.ylabel("Total Count")
plt.show()
labels_train = to_categorical(labels_train, num_classes = 10)
print(labels_train[0])
# Split the train data set into train and test

X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size = 0.1)
digitNet = Sequential()

digitNet.add(BatchNormalization(input_shape = (28, 28, 1)))
digitNet.add(Conv2D(filters = 16, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'same'))
digitNet.add(MaxPool2D(pool_size = 2))
digitNet.add(BatchNormalization())

digitNet.add(Conv2D(filters = 32, kernel_size = 3, kernel_initializer= 'he_normal', activation = 'relu', padding = 'same'))
digitNet.add(MaxPool2D(pool_size = 2))
digitNet.add(BatchNormalization())

digitNet.add(Conv2D(filters = 64, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'same'))
digitNet.add(MaxPool2D(pool_size = 2))
digitNet.add(BatchNormalization())

digitNet.add(Conv2D(filters = 128, kernel_size = 3, kernel_initializer = 'he_normal', activation = 'relu', padding = 'same'))
digitNet.add(MaxPool2D(pool_size = 2))
digitNet.add(BatchNormalization())

digitNet.add(GlobalAveragePooling2D())

digitNet.add(Dense(10, activation = 'softmax'))
digitNet.summary()
# Compile the model
digitNet.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Lets use a Data augmentor

datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range = 0.1,
        rotation_range=10,
        horizontal_flip = False,
        vertical_flip = False,
)

datagen.fit(X_train)

checkpointer = ModelCheckpoint(filepath = 'bestModel.hdf5', 
                               verbose=1, save_best_only = True)

reduce = LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)

digitNet.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), 
                       steps_per_epoch = X_train.shape[0] // 32, 
          validation_data = (X_valid, y_valid), epochs = 64,
          callbacks=[checkpointer, reduce], verbose=1)
digitNet.load_weights("bestModel.hdf5")
# Lets plot a confusion matrix
validationPredictions = digitNet.predict(X_valid)
v = np.argmax(validationPredictions, axis = 1) 
a = np.argmax(y_valid, axis = 1) 

cm = confusion_matrix(a, v)
cm_df = pd.DataFrame(cm, index = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
sns.heatmap(cm_df, annot = True, cmap = 'RdYlGn', linewidths = 0.2)
fig = plt.gcf()
fig.set_size_inches(12, 12)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
results = digitNet.predict(test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
print("Done")