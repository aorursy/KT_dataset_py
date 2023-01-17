# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train.shape

train.head()
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.shape
Y_train = train["label"]

X_train = train.drop(labels=["label"],axis = 1)
plt.figure(figsize = (16,8))

sns.countplot(Y_train, palette = "icefire")

plt.title("Number of digit classes")
img = X_train.iloc[5].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap="gray")

plt.title(train.iloc[0,0])

plt.axis("off")

plt.show()
# Normalize

X_train = X_train / 255.0

test = test / 255.0

X_train.shape,test.shape
# Reshape

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

X_train.shape,test.shape
# Label encoding

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes= 10)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)
from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = "Same",

                 activation = "relu", input_shape = (28,28,1)))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = "Same",

                 activation = "relu"))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = "Same",

                 activation = "relu"))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = "Same",

                 activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Dropout(0.20))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = "Same",

                 activation = "relu"))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = "Same",

                 activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Dropout(0.20))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = Adam(lr = 0.001, beta_1=0.9, beta_2 = 0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy",metrics = ["accuracy"])
epochs = 30

batch_size = 50
# data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False,  

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False, 

        rotation_range=0.2,  

        zoom_range = 0.2, 

        width_shift_range=0.2, 

        height_shift_range=0.2,

        horizontal_flip=False, 

        vertical_flip=False)  



datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)