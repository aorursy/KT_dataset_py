# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Activation, AveragePooling2D, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop,SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
train_csv = pd.read_csv("/kaggle/input/emnist/train.csv", index_col="id")
train_csv.head()
train_csv.digit.value_counts()
train_csv.info()
train_csv.describe()
train_csv.isnull().any().describe()
# split into X and y
train = train_csv.copy()
data = train.drop(["digit"], axis=1)
label = train_csv['digit']
# split into train and validation set
(train_X, valid_X, train_y, valid_y) = train_test_split(data, label, stratify=label, test_size=0.25, random_state=42)

letter_train = train_X["letter"]
train_X = train_X.drop(["letter"], axis=1)

letter_test = valid_X["letter"]
valid_X = valid_X.drop(["letter"], axis=1)

# normalize
train_X = train_X / 4.0
valid_X = valid_X / 4.0

#reshape
train_X = train_X.values.reshape(-1, 28, 28, 1)
valid_X = valid_X.values.reshape(-1, 28, 28, 1)
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
valid_y = lb.fit_transform(valid_y)
datagen = ImageDataGenerator(
        rotation_range=15,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range = 0.15)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.summary()
sgd = SGD(learning_rate=0.005)
rms = RMSprop(learning_rate=0.001)
adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, metrics=["accuracy"], loss="categorical_crossentropy")
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)
H = model.fit_generator(datagen.flow(train_X,train_y, batch_size=128),
                        epochs = 400, validation_data = (valid_X,valid_y),
                        verbose = 2, steps_per_epoch=train_X.shape[0] // 128,
                        callbacks=[learning_rate_reduction])
#plot
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 15))
plt.plot(range(400), H.history["loss"], label="train_loss")
plt.plot(range(400), H.history["accuracy"], label="val_loss")
plt.plot(range(400), H.history["val_loss"], label="train_acc")
plt.plot(range(400), H.history["val_accuracy"], label="val_acc")
plt.xlabel('# epochs')
plt.legend()
plt.show()
test_csv = pd.read_csv("/kaggle/input/emnist/test.csv")
test_csv
# split into X and y
test = test_csv.copy()
test_X = test.drop(["id", "letter"], axis=1)

# normalize
test_X = test_X / 4.0

#reshape
test_X = test_X.values.reshape(-1, 28, 28, 1)
