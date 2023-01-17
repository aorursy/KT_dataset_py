# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

from keras.models import Sequential

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, SGD, RMSprop

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import mean_absolute_error

from scipy.stats import zscore



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
y = np.array(pd.read_csv('../input/digit-recognizer/train.csv')).T[0]

X = pd.read_csv('../input/digit-recognizer/train.csv').drop('label', axis=1)

test = pd.read_csv('../input/digit-recognizer/test.csv')



print('target shape: ' + str(y.shape))

print('train shape:  ' + str(X.shape))

print('test shape:   ' + str(test.shape))
X
print(test)
print(y)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = np.array(X_train).reshape(33600, 28, 28, 1)

X_test = np.array(X_test).reshape(8400, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)

y_test = to_categorical(y_test, num_classes=10)

test = np.array(test).reshape(28000, 28, 28, 1)
X_train = X_train/255.0

X_test = X_test/255.0

test = test/255.0
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))



model.compile(loss='mean_absolute_error', metrics=['accuracy'], optimizer=Adam())
idg = ImageDataGenerator()

train_gen = idg.flow(X_train, y_train, batch_size=64)

test_gen = idg.flow(X_test, y_test, batch_size=64)
status = model.fit_generator(generator=train_gen, validation_data=test_gen, steps_per_epoch=1000, validation_steps=1000, epochs=10)
plt.plot(status.history['loss'], c='blue', label='Train')

plt.plot(status.history['val_loss'], c='red', label='Validation')

plt.title('Loss after epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.plot(status.history['accuracy'], c='blue', label='Train')

plt.plot(status.history['val_accuracy'], c='red', label='Validation')

plt.title('Accuracy after epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
ID = pd.read_csv(f'../input/digit-recognizer/sample_submission.csv')

ID = np.array(ID['ImageId'])

label = model.predict_classes(test)

submission = pd.DataFrame({'ImageId':ID, 'Label': label})
submission
submission.to_csv('submission.csv', index=False)