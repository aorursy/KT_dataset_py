import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pickle



labels = {'NORMAL': 0, 'PNEUMONIA': 1}

PATH = PATH = '/kaggle/input/chest-xray-pneumonia/chest_xray/'

IMG_SIZE = 80
def prepareData(dir_):

    X = []

    y = []

    

    for category in ['NORMAL', 'PNEUMONIA']:

        loc = os.path.join(PATH, dir_, category)

        for name in os.listdir(loc):

            img = cv2.imread(os.path.join(loc, name), 0)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(img)

            y.append(labels[category])

    return X, y
X, y = prepareData('train')

file = open('X_train.pickle', 'wb')

pickle.dump([X, y], file)

file.close()
with open('X_train.pickle', 'rb') as file:

    X, y = pickle.load(file)



# Feature Scaling

X = np.array(X) / 255.0
# Shuffling the Dataset

from sklearn.utils import shuffle



X, y = shuffle(X, y, random_state=4)

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D



model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
history = model.fit(X, np.array(y), batch_size=32, epochs=5, validation_split=0.1)
# Saving the weights of the model

model.save_weights('model.h5')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))



ax1.plot(history.history['loss'], c='r', label='Training Loss')

ax1.plot(history.history['val_loss'], c='b', label='Validation Loss')

ax1.set_xticks(np.arange(1, 5, 1))



ax2.plot(history.history['accuracy'], color='r', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='b',label="Validation accuracy")

ax2.set_xticks(np.arange(1, 5, 1))

ax2.set_yticks(np.arange(0, 1, 0.1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
# Preparing data for testing

X, y = prepareData('test')

file = open('X_test.pickle', 'wb')

pickle.dump([X, y], file)

file.close()
with open('X_test.pickle', 'rb') as file:

    X_test, y_test = pickle.load(file)



# Feature Scaling

X_test = np.array(X_test) / 255.0

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# Predicting from the trained model

predictions = model.predict(X_test)
y_pred = [list(row).index(max(list(row))) for row in predictions]

print('Test Accuracy:- ', 100.0*(1 - sum(abs(np.array(y_pred) - np.array(y_test))) / len(y_test)))
import pandas as pd



save = pd.DataFrame()

save['Pneumonia'] = np.array(y_pred)

save.to_csv('predictions.csv', index=False)