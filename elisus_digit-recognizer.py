# Let's keep all the imports we need here

# data analysis and wrangling

import pandas as pd

import numpy as np

# preprocessing

from keras.utils.np_utils import to_categorical

# model

from tensorflow.keras.models import Sequential

# layers

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
# Suppress future warnings

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# Let's read in the data first

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

train.head()
# We know this is a multiclass classification problem, but let's see how many classes there are

train.label.unique()
# Let's separate the pixels from the labels

X_train = (train.iloc[:,1:].values).astype('float32')

y_train = train.iloc[:,0].values.astype('int32')

# The test set has no labels

X_test = test.values.astype('float32')
# Normalize the data so the range goes from 0-255, to 0-1

X_train = X_train / 255.0

X_test = X_test / 255.0
# Reshape the array so each image is 28 by 28 by 1

# The last dimension is the pixel value

X_train = X_train.reshape(-1, 28, 28,1)

X_test = X_test.reshape(-1,28,28,1)
y_train = to_categorical(y_train)
# Set aside some data for validation

from sklearn.model_selection import train_test_split

X = X_train

y = y_train

# Validation set will be 10% of the training data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
model = Sequential()

model.add(Conv2D(28, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(28, 28, 1)))



model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(X_train, y_train,

          batch_size=64,

          epochs=6,

          validation_data=(X_val, y_val))
model2 = Sequential()

model2.add(Conv2D(28, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(28, 28, 1)))



model2.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))

model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Dropout(0.25))

model2.add(Flatten())

model2.add(Dense(256, activation='relu'))

model2.add(Dropout(0.5))

model2.add(Dense(10, activation='softmax'))

model2.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model2.fit(X_train, y_train,

          batch_size=64,

          epochs=6,

          validation_data=(X_val, y_val))
pred = model2.predict_classes(X_test, verbose=0)

submission = pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),

                         "Label": pred})

submission.to_csv("submission.csv", index=False, header=True)