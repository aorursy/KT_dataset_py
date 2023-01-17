import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import os

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, BatchNormalization

from keras import regularizers



from sklearn.utils import shuffle

print(os.listdir("../input"))
#Load CSV file into pandas

train_df = pd.read_csv('../input/train.csv')

train_df.head()
print("Prepping Data...")

image_directory = '../input/train/train/'



#Loading images and labels

X_train = [cv2.imread(image_directory + filename) for filename in os.listdir(image_directory)]

y_train = [train_df[train_df['id'] == filename].has_cactus.values for filename in os.listdir(image_directory)]



X_train = np.array(X_train)

y_train = np.array(y_train)

y_train = y_train.flatten()



X_train, y_train = shuffle(X_train, y_train, random_state = 2019)



print("Complete!")

print("X_train shape = {}".format(X_train.shape))

print("y_train shape = {}".format(y_train.shape))
#Show some random images

labels = ['No Cactus', 'Yes Cactus']



plt.figure(figsize=(10,10))

plt.subplot(1,3,1)

plt.title(labels[y_train[50]])

plt.imshow(X_train[50], interpolation='bilinear')



plt.subplot(1,3,2)

plt.title(labels[y_train[57]])

plt.imshow(X_train[57], interpolation='bilinear')



plt.subplot(1,3,3)

plt.title(labels[y_train[60]])

plt.imshow(X_train[60], interpolation='bilinear')
X_train = (X_train - X_train.mean()) / X_train.std()
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = (32,32,3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=5, validation_split = 0.25)
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = (32,32,3), activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=5, validation_split = 0.25)
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = (32,32,3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=5)
#Prep the testing data

print("Prepping Testing Data...")

testing_directory = '../input/test/test/'

X_test = [cv2.imread(testing_directory + filename) for filename in os.listdir(testing_directory)]

X_test = np.array(X_test)

print("Complete!")

print("X_test shape = {}".format(X_test.shape))
#Making Predictions

predictions = model.predict_classes(X_test)

submission = pd.read_csv('../input/sample_submission.csv')

submission['has_cactus'] = predictions

submission.sample(5)
submission.to_csv('./submissions.csv', header=False)