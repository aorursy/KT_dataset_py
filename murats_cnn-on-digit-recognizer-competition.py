# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
train.label.value_counts()
Y_train = train["label"]

X_train = train.drop(labels = ["label"], axis = 1)
Y_train.head()
X_train.head()
plt.figure(figsize=(15,7))

sns.countplot(Y_train, palette = "icefire")

plt.title("Number of digit classes")

Y_train.value_counts()
img = X_train.iloc[3].as_matrix()

img = img.reshape((28,28))

plt.imshow(img)

plt.title(train.iloc[3,0])

plt.axis("off")

plt.show()
img = X_train.iloc[0].as_matrix()

img = img.reshape((28,28))

plt.imshow(img, cmap = "gray")

plt.title(train.iloc[0,0])

plt.axis("off")

plt.show()
X_train = X_train / 255.0

test = test / 255.0

print("x train shape : ", X_train.shape)

print("test shape : ", test.shape)
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1, 28, 28, 1)

print("x_train shape : ", X_train.shape)

print("test shape : ", test.shape)
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size =0.1, random_state = 2)

print("x_train shape", X_train.shape)

print("x_val shape", X_val.shape)

print("y_train shape", Y_train.shape)

print("y_val shape", Y_val.shape)
#implementing with keras

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(filters = 8,

                kernel_size = (5,5),

                padding = "Same",

                activation = "relu",

                input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, 

                 kernel_size = (3,3),

                 padding = "Same",

                 activation = "relu"))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation = "softmax"))
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics =["accuracy"])
epochs = 3

batch_size = 250
datagen = ImageDataGenerator(

featurewise_center = False,

samplewise_center = False,

featurewise_std_normalization = False,

samplewise_std_normalization = False,

zca_whitening = False,

rotation_range = 0.5,

width_shift_range = 0.5,

height_shift_range = 0.5,

horizontal_flip = False,

vertical_flip = False)



datagen.fit(X_train) 
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),

                             epochs = epochs,

                              validation_data = (X_val,Y_val),

                              steps_per_epoch = X_train.shape[0])
plt.plot(history.history["val_loss"], color = "b", label = "validation loss")

plt.title("Test Loss")

plt.xlabel("Number of epochs")

plt.ylabel("loss")

plt.legend()

plt.show()
import seaborn as sns

y_pred = model.predict(X_val)

y_pred_classes = np.argmax(y_pred, axis = 1)

y_true = np.argmax(Y_val, axis= 1)

confusion_mtx = confusion_matrix(y_true, y_pred_classes)

f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(confusion_mtx, annot = True,

            linewidths = 0.01,

            cmap = "Greens",

            linecolor = "gray",

            fmt = ".1f",

            ax=ax)