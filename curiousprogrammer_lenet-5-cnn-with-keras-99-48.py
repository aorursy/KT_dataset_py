# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras 

from keras.models import Sequential

from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

X_train = df_train.iloc[:, 1:]

Y_train = df_train.iloc[:, 0]
X_train.head()
Y_train.head()
X_train = np.array(X_train)

Y_train = np.array(Y_train)
# Normalize inputs

X_train = X_train / 255.0
def plot_digits(X, Y):

    for i in range(20):

        plt.subplot(5, 4, i+1)

        plt.tight_layout()

        plt.imshow(X[i].reshape(28, 28), cmap='gray')

        plt.title('Digit:{}'.format(Y[i]))

        plt.xticks([])

        plt.yticks([])

    plt.show()
plot_digits(X_train, Y_train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(Y_train)

ax.set_title('Distribution of Digits', fontsize=14)

ax.set_xlabel('Digits', fontsize=12)

ax.set_ylabel('Count', fontsize=14)

plt.show()
#Train-Test Split

X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True, random_state=2019)

T_dev = pd.get_dummies(Y_dev).values

T_val = pd.get_dummies(Y_val).values
#Reshape the input 

X_dev = X_dev.reshape(X_dev.shape[0], 28, 28, 1)

X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPool2D(strides=2))

model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))

model.add(MaxPool2D(strides=2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(84, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.build()

model.summary()
adam = Adam(lr=5e-4)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
# Set a learning rate annealer

reduce_lr = ReduceLROnPlateau(monitor='val_acc', 

                                patience=3, 

                                verbose=1, 

                                factor=0.2, 

                                min_lr=1e-6)
# Data Augmentation

datagen = ImageDataGenerator(

            rotation_range=10, 

            width_shift_range=0.1, 

            height_shift_range=0.1, 

            zoom_range=0.1)

datagen.fit(X_dev)
model.fit_generator(datagen.flow(X_dev, T_dev, batch_size=100), steps_per_epoch=len(X_dev)/100, 

                    epochs=30, validation_data=(X_val, T_val), callbacks=[reduce_lr])
score = model.evaluate(X_val, T_val, batch_size=32)
score
df_test = pd.read_csv('../input/test.csv')

X_test = np.array(df_test)

X_test = X_test/255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

Y_test = model.predict(X_test)
Y_test = np.argmax(Y_test, axis=1)

Y_test[:5]
df_out = pd.read_csv('../input/sample_submission.csv')

df_out['Label'] = Y_test

df_out.head()
df_out.to_csv('out.csv', index=False)