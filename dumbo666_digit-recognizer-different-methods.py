import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.listdir('../input')
data = pd.read_csv('../input/train.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
X_train, X_test, y_train, y_test = train_test_split(normalize(data.values[:, 1:]), data.values[:, 0],
                                                   test_size=0.33, shuffle=True, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=137)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
test = pd.read_csv('../input/test.csv')
test.shape
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, AvgPool2D
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(height_shift_range=5, rotation_range=10, width_shift_range=5, zoom_range=0.1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(AvgPool2D())
model.add(Dropout(0.5))


model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AvgPool2D())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import normalize

X_train_ = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_ = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_valid_ = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
y_valid_cat = to_categorical(y_valid)

hist = model.fit_generator(datagen.flow(X_train_, y_train_cat, batch_size=64),
                  epochs=15, verbose=1, validation_data=(X_valid_, y_valid_cat),
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True, monitor='val_acc', baseline=0.95)])
score = model.evaluate(X_test_, y_test_cat, batch_size=32)
score
test_normed = normalize(test)
test_normed = test_normed.reshape(test_normed.shape[0], 28, 28, 1)

y_pred = model.predict(test_normed)
submission = pd.DataFrame(np.argmax(y_pred, axis=1), columns=['Label'])
submission.index += 1
submission.tail()
submission.to_csv('./submission.csv', index_label='ImageId', columns=['Label'])