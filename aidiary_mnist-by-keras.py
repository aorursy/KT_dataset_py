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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
test.head()
Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
del train
g = sns.countplot(Y_train)
Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train.iloc[0].min(), X_train.iloc[0].max()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
print(X_train.values.shape)
print(test.values.shape)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

print(X_train.shape)
print(test.shape)
Y_train = to_categorical(Y_train, num_classes=10)
print(Y_train.shape)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)
g = plt.imshow(X_train[0][:, :, 0])
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,  # 3 epochs
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
epochs = 30
batch_size = 86
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(X_val, Y_val),
                              verbose=2,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])
history.history.keys()
fig, ax = plt.subplots(2, 1)

# loss
ax[0].plot(history.history['loss'], color='b', label='Training loss')
ax[0].plot(history.history['val_loss'], color='r', label='validation_loss', axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

# accuracy
ax[1].plot(history.history['accuracy'], color='b', label='Training accuracy')
ax[1].plot(history.history['val_accuracy'], color='r', label='validation_accuracy', axes=ax[1])
legend = ax[1].legend(loc='best', shadow=True)
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)
cm = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(cm, classes=range(10))
print(Y_pred_classes, Y_pred_classes.shape)
print(Y_true, Y_true.shape)
errors = (Y_pred_classes - Y_true) != 0
print(errors)
print(np.sum(errors))
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]
print(Y_pred_classes_errors.shape)
print(Y_pred_errors.shape)
print(Y_true_errors.shape)
print(X_val_errors.shape)
nrows = 5
ncols = 6
fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12, 10))

for row in range(nrows):
    for col in range(ncols):
        n = row * ncols + col
        img = X_val_errors[n, :, :, 0]
        ax[row, col].imshow(img)
        ax[row, col].set_title('{}->{}'.format(Y_true_errors[n], Y_pred_classes_errors[n]))
        ax[row, col].axis('off')
results = model.predict(test)
results.shape
results = np.argmax(results, axis=1)
results.shape
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.head()
submission.to_csv('cnn_mnist_augmentation.csv', index=False)