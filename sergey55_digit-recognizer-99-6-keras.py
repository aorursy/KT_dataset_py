# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
num_classes = 10
img_rows, img_cols = 28, 28

def prep_data(raw):
    out_y = to_categorical(raw.label, num_classes)
    
    x = raw.values[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

def prep_test_data(raw):
    x = raw.values[:, :]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    return out_x / 255 

train_file = '/kaggle/input/digit-recognizer/train.csv';
data = pd.read_csv(train_file)

X, y = prep_data(data)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)

test_file = '/kaggle/input/digit-recognizer/test.csv';
test_data = pd.read_csv(test_file)

X_test = prep_test_data(test_data)
batch_size = 128

data_generator_with_aug = ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,                                             
                                             rotation_range = 10,
                                             zoom_range = 0.1,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             horizontal_flip=False,
                                             vertical_flip=False)

data_generator_without_aug = ImageDataGenerator()

train_generator_with_aug = data_generator_with_aug.flow(X_train, y_train, 
                                                        batch_size = batch_size)

train_generator_without_aug = data_generator_without_aug.flow(X_train, y_train)

steps_per_epoch = int(X_train.shape[0] / batch_size)
lrr = ReduceLROnPlateau(monitor='val_accuracy',
                        patience=3,
                        verbose=1,
                        factor=0.5,
                        min_lr=0.000001)
model = Sequential()

model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(img_rows, img_cols, 1), padding='Same'))
model.add(Conv2D(32, kernel_size=5, activation='relu', padding='Same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='Same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='Same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#optimizer = Adam(lr=5e-3)
optimizer2 = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=optimizer2, metrics=['accuracy'])
#Learn model using original data
history1 = model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val), callbacks=[lrr])

#optimizer.learning_rate.assign(5e-4)

#Learn existing model with augmentation
history2 = model.fit_generator(train_generator_with_aug, epochs=70, validation_data=(X_val, y_val), steps_per_epoch=steps_per_epoch, callbacks=[lrr])
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
y_pred = model.predict(X_val).argmax(axis = 1)
Y_true = y_val.argmax(axis=1)
conf_matrix = confusion_matrix(Y_true, y_pred)

plot_confusion_matrix(conf_matrix, classes=range(10))
predictions = model.predict(X_test)

predicted_classes = [p.argmax() for p in predictions]

submission = pd.DataFrame(predicted_classes, columns=['Label'])
submission.index.name = 'ImageId'
submission.index += 1

submission.to_csv('submit.csv', index=True)
