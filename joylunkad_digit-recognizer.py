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

from keras.utils.np_utils import to_categorical

%matplotlib inline
df_train = pd.read_csv("../input/digit-recognizer/train.csv")

df_test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = df_train['label']

X_train = df_train.drop(labels = 'label', axis = 1).values

X_test = df_test/255.0

X_train = X_train/255.0
fig = plt.figure(figsize=(20,20))

for i in range(6):

    ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])

    ax.imshow(X_train[i].reshape(28,28), cmap='gray')

    ax.set_title(str(Y_train[i]))
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
#one hot

Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 7)
print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D((2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D((2,2), strides = (2,2)))

model.add(Dropout(0.25))

          

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(10, activation = 'softmax'))



model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_data = (X_val, Y_val)) 
results = model.predict(X_test)
results = np.argmax(results, axis = 1)

results = pd.Series(results, name  = 'Label')

results.head(10)
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)

submission.to_csv("submission.csv", index = False)