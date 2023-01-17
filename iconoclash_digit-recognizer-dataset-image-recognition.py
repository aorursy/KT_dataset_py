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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
digit = pd.read_csv("../input/train.csv")
digit.head()
# based on above information we know that this dataset have 5 rows × 785 columns ,because there are too much columns so usual detailed info must not be used
#count the nan values in the dataframe

sum(digit.isna().sum())
sns.countplot(x="label",data=digit,palette=sns.color_palette("Set1"))
# reshape the pixel in the features
img = digit[digit["label"]==0].iloc[2,1:].values.reshape(28,28)
plt.imshow(img, cmap="viridis")
# reshape the pixel in the features
img = digit[digit["label"]==1].iloc[5,1:].values.reshape(28,28)
plt.imshow(img, cmap="viridis")
# reshape the pixel in the features
img = digit[digit["label"]==2].iloc[1,1:].values.reshape(28,28)
plt.imshow(img, cmap="viridis")
fig, ax = plt.subplots()
plt.subplot(251)
img = digit[digit["label"]==1].iloc[1,1:].values.reshape(28,28)
plt.imshow(img, cmap="viridis")

plt.subplot(252)
img = digit[digit["label"]==2].iloc[1,1:].values.reshape(28,28)
plt.imshow(img, cmap="viridis")

plt.subplot(257)
img = digit[digit["label"]==3].iloc[1,1:].values.reshape(28,28)
plt.imshow(img, cmap="viridis")
fig, ax = plt.subplots()

for a,b in zip(range(1,11),range(0,10)):
    plt.subplot(2,5,a)
    img = digit[digit["label"]==b].iloc[1,1:].values.reshape(28,28)
    plt.imshow(img, cmap="viridis")
feature = digit.drop(["label"],axis=1)
target=digit["label"]
# from sklearn.model_selection import train_test_split,cross_val_score
# X_train, X_test, y_train, y_test = train_test_split(feature,target,
#                                                     test_size=0.20,random_state=101)
# X_train.shape
# 'float' object cannot be interpreted as an integer
# param1 = int(X_train.shape[0])
# int(X_train.shape[0])
# param23 = int(X_train.shape[1]**(1/2))
# int(X_train.shape[1]**(1/2))
# X_train = X_train.values.reshape(param1, param23,param23,1)
# X_train.shape
# input_shape= X_train.shape[1:]
# X_train.shape[1:]
#Using TensorFlow backend.

# from tensorflow.python import keras
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense
# model = Sequential()

# model.add(Conv2D(filters=28, kernel_size=(7,7),input_shape = input_shape))
# model.add(Conv2D(28, kernel_size=(5,5), activation="relu"))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(28, kernel_size=(5,5), activation="relu"))
# model.add(Conv2D(28, kernel_size=(3,3), activation="relu"))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())

# model.add(Dense(112, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation="softmax"))

# model.compile(loss="categorical_crossentropy",
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])
# model.summary()
# from keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train, num_classes = 10)
# from tensorflow.python.keras.callbacks import EarlyStopping
# model.fit(X_train, y_train,
#           batch_size=28*3,
#           epochs=5,
#           validation_split=0.2,
#           callbacks=[EarlyStopping(patience=1)],
#           verbose=1)
# X_test.shape
# 'float' object cannot be interpreted as an integer
# param1 = int(X_test.shape[0])
# int(X_test.shape[0])
# param23 = int(X_test.shape[1]**(1/2))
# int(X_test.shape[1]**(1/2))
# X_test = X_test.values.reshape(param1, param23,param23,1)
# X_test.shape
# y_test = to_categorical(y_test, num_classes = 10)
# y_test.shape
# y_pred = model.predict(X_test)
feature.shape
# 'float' object cannot be interpreted as an integer
param1 = int(feature.shape[0])
int(feature.shape[0])
param23 = int(feature.shape[1]**(1/2))
int(feature.shape[1]**(1/2))
feature = feature.values.reshape(param1, param23,param23,1)
feature.shape
input_shape= feature.shape[1:]
feature.shape[1:]
#Using TensorFlow backend.

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
target = to_categorical(target, num_classes = 10)
from tensorflow.python.keras.callbacks import EarlyStopping
model = Sequential()

model.add(Conv2D(filters=28, kernel_size=(7,7),input_shape = input_shape))
model.add(Conv2D(28, kernel_size=(5,5), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(28, kernel_size=(5,5), activation="relu"))
model.add(Conv2D(28, kernel_size=(3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(112, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="Adam",
              metrics=['accuracy'])
model.summary()
model.fit(feature, target,
          batch_size=28*10, #minimum ideal batch size
          epochs=(28*10),
          validation_split=0.2,
          callbacks=[EarlyStopping(patience=3)],
          verbose=1)
# digit_test = pd.read_csv("../input/test.csv")
# digit_test.head()
# digit_sample = pd.read_csv("../input/test.csv")