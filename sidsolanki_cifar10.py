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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import random
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
classes = ['Airplanes', 'Cars', 'Birds', 'Cats', 'Deers',
               'Dogs', 'Frogs', 'Horses', 'Ships', 'Trucks']
num_classes = 10
n_images = X_train.shape[0]
img_rows, img_cols, channels = 32, 32, 3
# inspect first image
i = 0

plt.figure()
plt.imshow(X_train[i])
plt.colorbar()
plt.grid(False)
plt.title(classes[y_train[i][0]])
plt.show()
# inspect random 25 images

randomlist = random.sample(range(0, n_images), 25)

plt.figure(figsize = (20,20))
for i in range(len(randomlist)):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[randomlist[i]])
    plt.xlabel(classes[y_train[randomlist[i]][0]])
plt.show()
## Check Target

y_train_df = pd.DataFrame(y_train)
y_test_df  = pd.DataFrame(y_test)

fig, ax = plt.subplots(1,2)
sns.countplot(y = 0, data = y_train_df, ax=ax[0])			# along y-axis # sort by count
sns.countplot(y = 0, data = y_test_df, ax=ax[1])
fig.show()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test  = X_test / 255.0
print(X_train.shape)
print(X_test.shape)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)
model = Sequential()
model.add(Input (shape = (img_rows, img_cols, channels)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense (256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense (num_classes, activation = 'softmax'))
model.summary()
model.compile(loss ='categorical_crossentropy', 
              optimizer = 'Adam', 
              metrics =['accuracy'])
cp = ModelCheckpoint('./best_model.h5', 
                     monitor='val_loss', 
                     mode='min',
                     save_best_only=True,
                     verbose = 1)

es = EarlyStopping(monitor='val_loss', 
                   verbose=1, 
                   patience=5,
                   min_delta=0,
                   restore_best_weights = True)

callbacks = [cp, es]
epochs = 100
batch_size = 32
%%time
history = model.fit(X_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    callbacks = callbacks,
                    validation_split = 0.2,
                    verbose = 1)
history.history.keys()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(X_test.shape)
plt.figure(figsize = (20,20))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    plt.title('Prediction: {}'.format(classes[y_pred[i]]))
    plt.xlabel('True: {}'.format(classes[y_test[i]]))
plt.show()
### Confusion Matrix ###

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True)
plt.show()
### Accuracy ###

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

print(f'accuracy : {acc:.2f}%')

### Classification report ###

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))