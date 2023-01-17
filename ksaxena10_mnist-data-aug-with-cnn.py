# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install tensorflow-gpu

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
train_data_path = '/kaggle/input/digit-recognizer/train.csv'

test_data_path = '/kaggle/input/digit-recognizer/test.csv'
train = pd.read_csv(train_data_path)

test = pd.read_csv(test_data_path)
train.head()
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(Y_train)



Y_train.value_counts()
# Check the data

X_train.isnull().any().describe()
test.isnull().any().describe()

# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
# Some examples

g = plt.imshow(X_train[0][:,:,0])
model = keras.models.Sequential()

model.add(Conv2D(32, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
# Define the optimizer

from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
for layer in model.layers:

    print(layer.get_output_at(0).get_shape().as_list())
# Set a learning rate annealer

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
## consider using this for more variety

from keras.preprocessing.image import ImageDataGenerator

data_generator_with_aug = keras.preprocessing.image.ImageDataGenerator(validation_split=.2, width_shift_range=.1,

                                                                       height_shift_range=.1, rotation_range=20,

                                                                       zoom_range=.1, shear_range=.1)

data_generator_with_aug.fit(X_train)
# Fit the model

epochs = 5

batch_size = 86

history = model.fit_generator(data_generator_with_aug.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)