import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, Model

from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, BatchNormalization

from keras.optimizers import Adam

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score



np.set_printoptions(suppress=True)
raw_data = np.loadtxt('../input/kepler-labelled-time-series-data/exoTrain.csv', skiprows=1, delimiter=',')

x_train = raw_data[:, 1:]

y_train = raw_data[:, 0, np.newaxis] - 1

raw_data = np.loadtxt('../input/kepler-labelled-time-series-data/exoTest.csv', skiprows=1, delimiter=',')

x_test = raw_data[:, 1:]

y_test = raw_data[:, 0, np.newaxis] - 1

del raw_data
x_train_positives = x_train[np.squeeze(y_train) == 1]

x_train_negatives = x_train[np.squeeze(y_train) == 0]
num_rotations = 100

for i in range(len(x_train_positives)):

     for r in range(num_rotations):

          rotated_row = np.roll(x_train[i,:], shift = r)

          x_train = np.vstack([x_train, rotated_row])
plt.plot(x_train_positives[0])

plt.plot(x_train_negatives[0])

plt.show()
x_train = x_train[:,0:1000]
y_train = np.vstack([y_train, np.array([1] * len(x_train_positives) * num_rotations).reshape(-1,1)])
x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / np.std(x_train, axis=1).reshape(-1,1))[:,:,np.newaxis] 

x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))[:,:,np.newaxis]
model = Sequential()

model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))

model.add(MaxPool1D(strides=2))

model.add(BatchNormalization())

model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))

model.add(MaxPool1D(strides=2))

model.add(BatchNormalization())

model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))

model.add(MaxPool1D(strides=2))

model.add(BatchNormalization())

model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))

model.add(MaxPool1D(strides=2))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=20)
non_idx = np.where(y_test[:,0] == 0.)[0]

yes_idx = np.where(y_test[:,0] == 1.)[0]

y_predict = model.predict(x_test[:,0:1000])
plt.plot([y_predict[i] for i in yes_idx], 'bo')

plt.show()

plt.plot([y_predict[i] for i in non_idx], 'ro')

plt.show()
threshold = 0.5

print(classification_report(y_test,y_predict >= threshold))