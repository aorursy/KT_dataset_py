# Imports

import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from keras.models import Sequential

from keras.layers import Dense
# Read the dataset

train_csv_df = pd.read_csv('../input/train.csv')

test_csv_df = pd.read_csv('../input/test.csv')



y_train = np.array(train_csv_df['label'])

X_train = np.array(train_csv_df.drop('label', 1))



X_test = np.array(test_csv_df)
# Plot few image from training dataset

count = 0

for index in range (0, 6):

  plt.subplot(2, 3, count + 1)

  plt.title(y_train[index])

  plt.imshow(X_train[index].reshape(28, 28), cmap="gray")

  count += 1

plt.show()
# Convert training set output to one hot encoding

y_train = np.eye(10)[y_train]



# Normalize the data

X_test = X_test / 255

X_train = X_train / 255
model = Sequential()

model.add(Dense(units = 512, input_dim = 784, activation = 'relu'))

model.add(Dense(units = 330, activation = 'relu'))

model.add(Dense(units = 212, activation = 'relu'))

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 152, activation = 'relu'))

model.add(Dense(units = 152, activation = 'relu'))

model.add(Dense(units = 10, activation = 'softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=50, batch_size=6000)
predictions = model.predict(X_test)



# Plot few image from predictions

count = 0

for index in range (0, 6):

  plt.subplot(2, 3, count + 1)

  plt.title(np.argmax(predictions[index]))

  plt.imshow(X_test[index].reshape(28, 28), cmap="gray")

  count += 1

plt.show()
with open ('output.csv', 'w+') as file:

  file.write('ImageId,Label\n')

  for index in range(X_test.shape[0]):

    file.write('{0},{1}\n'.format(index+1, np.argmax(predictions[index])))