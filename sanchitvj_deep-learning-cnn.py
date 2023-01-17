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
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam, RMSprop
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head(3)
test.head(3)
X = train.drop('label', axis = 1)

y = train['label']

X = X.values.reshape(-1, 28, 28, 1)

X = X / 255

y = to_categorical(y)

plt.imshow(X[1][:,:,0])

print(str(y[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)
datagen = ImageDataGenerator(zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 10) 
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides = (2,2)))

model.add(Dropout(0.4))



model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides = (2,2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 35, validation_data = (X_test, y_test), callbacks = [learning_rate_reduction])
loss = pd.DataFrame(model.history.history)

loss[['loss', 'val_loss']].plot()

loss[['accuracy', 'val_accuracy']].plot()
final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
test = test.values.reshape(-1, 28, 28, 1) / 255

y_pred = model.predict(test, batch_size = 64)



y_pred = np.argmax(y_pred,axis = 1)

y_pred = pd.Series(y_pred,name="Label")

y_pred
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred],axis = 1)

submission.to_csv("submission.csv",index=False)