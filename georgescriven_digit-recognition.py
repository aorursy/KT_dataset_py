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
from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten

from keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_images = (pd.read_csv("/kaggle/input/digit-recognizer/test.csv").values).astype('float32')
train_images = (train.iloc[:, 1:].values).astype('float32')

train_labels = train['label'].values.astype('int32')
train_images.shape
train_labels.shape
train_images = train_images.reshape(train_images.shape[0] ,28,28)



for i in range(0,5):

    plt.subplot(330+(i+1))

    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))

    plt.title(train_labels[i])
train_images = train_images.reshape(train_images.shape[0] ,28*28)
from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)

num_classes = train_labels.shape[1]

num_classes
plt.title(train_labels[8])

plt.plot(train_labels[8])

plt.xticks(range(10));
seed = 43

np.random.seed(seed)
train_images.shape
model = Sequential()

model.add(Dense(32, activation='relu', input_dim=(28*28)))

model.add(Dense(16, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=RMSprop(lr=0.001),

             loss='categorical_crossentropy',

             metrics=['accuracy'])
history=model.fit(train_images, train_labels, validation_split=0.05,

                 epochs=25, batch_size=64)
history_dict = history.history

history_dict.keys()
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')
acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.show()
model = Sequential()

model.add(Dense(64, activation='relu', input_dim=(28*28)))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(train_images, train_labels, epochs=15, batch_size=64)
predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({'ImageId':list(range(1,len(predictions) + 1)), "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)