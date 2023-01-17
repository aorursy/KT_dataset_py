# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler



train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/train.csv")
train.info()
train.head()
print(test.shape)
X_train = train.drop(labels = ["label"], axis = 1)
Y_train = train['label']
test = test.drop(labels = ["label"], axis = 1)

Y_train.value_counts()

sns.set(style = "darkgrid", palette = "RdBu")
plt.figure(figsize = (12,6))
sns.countplot(Y_train)
plt.title("Count of digit classes")
img = X_train.iloc[5].as_matrix()
img = img.reshape((28,28))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title(train.iloc[5,0])
plt.axis('off')
plt.show()
#checking for missing data
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train / 255.0
test = test / 255.0
print("shape of test: ", test.shape)
print("shape of train: ", X_train.shape)
# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

from keras.utils.np_utils import to_categorical # convert to one-hot encoding
Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_val.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_val.shape)
# Image example
plt.imshow(X_train[4][:,:,0],cmap='gray')
plt.show()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(filters = 8 , kernel_size = (5,5), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 16 , kernel_size = (5,5), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = (2,2), strides = None))
model.add(Dropout(0.5))

model.add(Conv2D(filters = 16 , kernel_size = (3,3), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 32 , kernel_size = (3,3), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size = (2,2), strides = None))
model.add(Dropout(0.5))




model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()


# define the optimizer
optimizer = Adam(learning_rate=0.001 )
# compile
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
epochs = 20
batch_size = 128

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), 
                              epochs = epochs,
                              validation_data = (X_val, Y_val),
                              steps_per_epoch = X_train.shape[0] // batch_size
                             )

plt.plot(history.history["val_loss"], color = 'g', label = "validation loss" )
plt.title("Loss function")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
from sklearn.metrics import confusion_matrix

Y_pred = model.predict(X_val)
Y_pred_class = np.argmax(Y_pred, axis = 1)
Y_true = np.argmax(Y_val, axis = 1)
confusion = confusion_matrix(Y_true, Y_pred_class)


f, ax = plt.subplots(figsize= (12,8))
sns.heatmap(confusion, annot = True, cmap = "YlGnBu", ax = ax, fmt= '.1f')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()
