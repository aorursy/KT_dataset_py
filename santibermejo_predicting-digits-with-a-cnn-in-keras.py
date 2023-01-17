import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

y = np.array(train_df['label'])
train_df.drop(columns='label', inplace = True)
def df_to_masksArray(df):
  imageRow = []

  for index, row in df.iterrows():
    imageRow.append(row.values)

  return imageRow
def df_to_masks(df):
  imageRow = []

  for index, row in df.iterrows():
    imageRow.append(row.values.reshape(28,28))

  return pd.DataFrame(pd.Series(imageRow))
train_masks = df_to_masks(train_df)
test_masks = df_to_masks(test_df)

train_masks_array = np.asarray(df_to_masksArray(train_df)).reshape(-1,28,28,1)
test_masks_array = np.asarray(df_to_masksArray(test_df)).reshape(-1,28,28,1)
# Print the images

from pylab import imshow, show, get_cmap
from random import *

f, axarray = plt.subplots(1,4)
for i in range(4):
  axarray[i].imshow(train_masks.iloc[np.random.randint(0,train_masks.shape[0])][0], interpolation='nearest')
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
# Initialising the CNN
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), activation = "relu", input_shape=(28, 28, 1)))
classifier.add(MaxPooling2D((2,2)))

classifier.add(Conv2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D((2,2)))

classifier.add(Conv2D(64, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D((2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = "relu"))

classifier.add(Dropout(0.5))

classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from sklearn.model_selection import train_test_split

y_train_categorical_labels = to_categorical(y, num_classes=10)
y_test_categorical_labels = to_categorical(y, num_classes=10)

classifier.fit(x = train_masks_array, y = y_train_categorical_labels, epochs = 50, batch_size = 100)
# Predicting the Test set results
y_pred = classifier.predict(test_masks_array)

y_pred = np.around(y_pred).argmax(1)
subs = pd.DataFrame()
subs['ImageId'] = np.asarray(range(1,28001))
subs['Label'] = y_pred

subs.to_csv('sample_submission.csv', index = False)