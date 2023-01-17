# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import cm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



def show_img(img):

    """

    Nicely show an image with matplotlib.

    """

    plt.figure(figsize=(10, 6))

    if len(img.shape) == 2:

        plt.imshow(img, cmap=cm.Greys_r)

    else:

        plt.imshow(img)

    plt.axis('off')

    plt.show()
train_df = pd.read_csv('../input/fashion-mnist_train.csv')

test_df = pd.read_csv('../input/fashion-mnist_test.csv')
train_x = train_df.iloc[:, 1:].values

test_x = test_df.iloc[:, 1:].values

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float64')

test_x = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float64')

train_x /= 255

test_x /= 255
train_y = train_df.iloc[:, 0].values

test_y = test_df.iloc[:, 0].values

print(test_y.shape)
import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
from keras.utils import to_categorical

#one-hot encode target column

train_y = to_categorical(train_y)

test_y = to_categorical(test_y)

print(train_y)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

#create model

model = Sequential()

#add model layers

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=3, padding="same", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model

result = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=15)
prediction = model.predict(test_x)
from sklearn import metrics

import seaborn as sns



cm = metrics.confusion_matrix(test_y.argmax(axis=1), prediction.argmax(axis=1))

cm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cm)

fig_corr, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g', cmap='Blues', ax=ax)

plt.show()
plt.plot(result.history['acc'])

plt.plot(result.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(result.history['loss'])

plt.plot(result.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()