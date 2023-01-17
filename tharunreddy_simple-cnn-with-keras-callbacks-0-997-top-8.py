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
import os



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



len(train_df), len(test_df)
# sample rows



print(train_df.head())

print(test_df.head())
28*28
# sample train images



np.random.seed(13)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

random_rows = np.random.choice(train_df.index, size=8, replace=False)

for ax, idx in zip(axes.flat, random_rows):

    row = train_df.iloc[idx, 1:].values

    img_matrix = row.reshape(28, 28)

    ax.imshow(img_matrix, cmap="gray")

    ax.set_title(train_df.iloc[idx, 0], fontdict={"fontsize": 24})
# sample test images



fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

np.random.seed(13)

random_rows = np.random.choice(test_df.index, size=4, replace=False)

for ax, idx in zip(axes.flat, random_rows):

    row = test_df.iloc[idx, :]

    ax.imshow(row.values.reshape(28, 28), cmap="gray")
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Activation

from keras.layers import Dropout, BatchNormalization

from keras.optimizers import Adam, RMSprop

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
# rescale inputs

train_df.iloc[:, 1:] /= 255

test_df /= 255



train_inputs = train_df.iloc[:, 1:].values.reshape(len(train_df), 28, 28, 1)

test_inputs = test_df.iloc[:, :].values.reshape(len(test_df), 28, 28, 1)



train_inputs.shape, test_inputs.shape
one_hot_labels = to_categorical(train_df.label.values, num_classes=10)

one_hot_labels[: 5]
# Split train - validation

X_train, X_val, y_train, y_val = train_test_split(train_inputs, one_hot_labels, test_size=0.2, random_state=13)
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), name="conv1", padding="same"))

model.add(Activation("relu", name="act1"))

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), name="conv2", padding="same"))

model.add(Activation("relu", name="act2"))

model.add(MaxPool2D((2, 2), name="mpool1"))

model.add(Dropout(0.25, name="drop1"))



model.add(Conv2D(64, (3, 3), name="conv3", padding="same"))

model.add(Activation("relu", name="act3"))

model.add(Conv2D(64, (3, 3), name="conv4", padding="same"))

model.add(Activation("relu", name="act4"))

model.add(MaxPool2D((2, 2), name="mpool2"))

model.add(Dropout(0.25, name="drop2"))



model.add(Flatten(name="flat1"))

model.add(BatchNormalization(name="bn1"))

model.add(Dense(512, activation="relu", name="dense1"))

model.add(Dropout(0.5, name="drop3"))



model.add(Dense(10, activation="softmax", name="softmax"))
model.summary()
# Using Keras 'ImageDataGenerator'. Function returns a generator with images transformed

datagen = ImageDataGenerator(zoom_range = 0.1,

                             height_shift_range = 0.1,

                             width_shift_range = 0.1,

                             rotation_range = 10)

datagen.fit(X_train)
# By using Keras callbacks, we can control the learning rate automatically while model is getting built

# Below ReduceLROnPlateau decreases learning rate based on change in validation accuracy

# patience - waits for 3 epochs

# min_lr - if learning rate is not changed by "min_lr" amount

# factor - existing learning rate by this value



lr_reduce = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.5, verbose=1, min_lr=0.00001)
model.compile(optimizer=Adam(), metrics=["accuracy"], loss="categorical_crossentropy")

model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),

                    validation_data=(X_val, y_val), epochs=100, callbacks=[lr_reduce], verbose=1)
# predictions

test_preds = model.predict(test_inputs)

test_labels = np.argmax(test_preds, axis=1)

test_labels[:5]
# submission dataframe

sub_df = pd.concat([pd.Series(range(1, 28001), name="ImageId"), pd.Series(test_labels, name="Label")], axis=1)



# save submission

sub_df.to_csv('submission.csv', index=False)