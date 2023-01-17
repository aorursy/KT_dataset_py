# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load Train & Test Data
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(df)
print(df.shape)

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
import matplotlib.pyplot as plt

# Change Image Size (Keras need this image size format)
Y = df['label']
X = df.drop(labels = ['label'], axis = 1).values.reshape(-1,28,28,1)

X = X/255
test = test/255
test = test.values.reshape(-1,28,28,1)
sns.countplot(Y)
Y.value_counts()
df.isnull().any().describe()

# Change Y to categorical
Y = to_categorical(Y)
# Try plot some images
plt.imshow(X[10][:,:,0])
# Split data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 7)
# Create a simple model that only contains of 1 Convolution Layer, 1 Max Pool Layer, 1 Dropout Layer
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=3,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_test, Y_test))
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Get the result for Sample Submission
x = model.predict(test)

# Select Index with Maximum Probability (Remember Python index start from 0)
test_result = np.argmax(x, axis = 1)
test_result = pd.Series(test_result, name = 'Label')
print(test.shape)
submission = pd.concat([pd.Series(range(1,28001), name = 'ImageID'), test_result], axis = 1)
print(submission)
submission.to_csv('/kaggle/working/sample_submission.csv', index = False)
