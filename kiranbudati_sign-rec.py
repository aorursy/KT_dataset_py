# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import random

import numpy as np



import sklearn

from sklearn.model_selection import train_test_split

import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
train_data = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")
train_data.head()
unique_labels = np.sort(train_data['label'].unique())
plt.imshow((train_data.iloc[0,1:].values).reshape(28,28))

plt.title(train_data.iloc[0,0])
fig, axes = plt.subplots(6,4,figsize=(14,14),sharex=True, sharey=True)

for i in range(len(unique_labels)):

    value = train_data.loc[train_data['label']==unique_labels[i]].iloc[0]

    subplot_row = i//4

    subplot_col = i%4

    ax = axes[subplot_row, subplot_col]

    plot_image = np.reshape(value[1:].values, (28,28))

    ax.imshow(plot_image, cmap='gray')    

    ax.set_title('Digit Label: {}'.format(unique_labels[i]))

    ax.axis('off')
X = train_data.iloc[:,1:].values/255

y = train_data.iloc[:,0].values



y = to_categorical(y)
X.shape,y.shape
X = X.reshape(X.shape[0],28,28,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# #create model

# model = Sequential()

# #add model layers

# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

# model.add(Conv2D(32, kernel_size=3, activation='relu'))

# model.add(Flatten())

# model.add(Dense(25, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(25, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15,batch_size=50)
plt.plot(h.history['accuracy'])

plt.plot(h.history['val_accuracy'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])



plt.show()
plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title("loss")

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['train','test'])



plt.show()