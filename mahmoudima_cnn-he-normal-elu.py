# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Lambda

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.regularizers import L1L2

dense_regularizer = L1L2(l2=0.0001)

from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.metrics import confusion_matrix

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)

training_dataset.head()
y = training_dataset['label'].copy()

X = training_dataset.drop(['label'], axis='columns')

del training_dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print('X train shape: {}'.format(X_train.shape))

print('y train shape: {}'.format(y_train.shape))

print('X test shape: {}'.format(X_test.shape))

print('y test shape: {}'.format(y_test.shape))
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



X_train = X_train.reshape(-1, 28, 28,1)

X_test = X_test.reshape(-1, 28, 28,1)



print('X train shape: {}'.format(X_train.shape))

print('y train shape: {}'.format(y_train.shape))

print('X test shape: {}'.format(X_test.shape))

print('y test shape: {}'.format(y_test.shape))
X_train = X_train / 255.

X_test = X_test / 255.
BATCH_SIZE = 128



def Model_1(x=None):

    # we initialize the model

    model = Sequential()



    # Conv Block 1

    model.add(Conv2D(64, (5, 5), input_shape=(28,28,1),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(64, (5, 5),   padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(64, (5, 5),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.3))



    # Conv Block 2

    model.add(Conv2D(128, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(128, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(128, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    # Conv Block 3

    model.add(Conv2D(256, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(256, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(256, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    # Conv Block 4

    model.add(Conv2D(512, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(Conv2D(512, (3, 3),  padding='same', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))



    # FC layers

    model.add(Flatten())

    model.add(Dense(10, activation='softmax', kernel_regularizer=dense_regularizer,kernel_initializer="he_normal"))



    return model



model = Model_1()

model.summary()
model.compile(Adam(lr=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
import tensorflow.keras

callbacks_list = [

tensorflow.keras.callbacks.EarlyStopping(

monitor='val_accuracy', min_delta=0.0001, 

patience=20, verbose=1, mode='auto',

baseline=None, restore_best_weights=True),

tensorflow.keras.callbacks.ReduceLROnPlateau(

monitor='val_accuracy',

factor=0.5,

patience=10,

verbose=1,

mode='auto'),

tensorflow.keras.callbacks.ModelCheckpoint(

filepath='./my_model.h5',

monitor='val_accuracy',

save_best_only=True,

)

]
history = model.fit(X_train, y_train, epochs=200, batch_size=BATCH_SIZE, verbose=1, validation_data=(X_test,  y_test), callbacks=callbacks_list)
metrics = pd.DataFrame(history.history)

metrics.head()
submission_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission_data.head()
submission_data = np.array(submission_data)

submission_data = submission_data.reshape(-1, 28, 28,1)

print(submission_data[0])
submission_data = submission_data / 255.

submission_classes = model.predict_classes(submission_data)

print(submission_classes[0])
submission_classes.shape
sub = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sub.head()
sample_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_sub.head()
ids = [i for i in range(1, sub.shape[0] + 1)]

submission = pd.DataFrame({'ImageId': ids, 'Label': submission_classes})

submission.head()
filename = 'CNN_He_Normal_eLU.csv'

submission.to_csv(filename, index=False)