# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_train = train['label']
X_train = train.drop('label', axis=1)

del train

# Show the distribution of label classes
g = sns.countplot(y_train)
y_train.value_counts()
# Normalize the data
X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.
X_test = test.values.reshape(-1, 28, 28, 1) / 255.

y_train = to_categorical(y_train, num_classes = 10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
print(X_train.shape, X_val.shape)
g = plt.imshow(X_train[0][:,:,0])
model = Sequential()

model.add(Conv2D(filters=32,
                kernel_size=(5,5),
                padding='SAME',
                activation='relu',
                input_shape=(28,28,1)))
model.add(Conv2D(filters=32,
                kernel_size=(5,5),
                padding='SAME',
                activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                kernel_size=(3,3),
                padding='SAME',
                activation='relu'))
model.add(Conv2D(filters=64,
                kernel_size=(3,3),
                padding="SAME",
                activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# from keras.utils import plot_model
# plot_model(model)

model.summary()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc',
    patience=3,
    factor=0.5,
    min_lr=0.00001
)
epochs = 1
batch_size = 86
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(X_train)
history = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=2,
    steps_per_epoch=X_train.shape[0],
    callbacks=[learning_rate_reduction]
)
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thres = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[0])):
        plt.text(j, i, cm[i,j],
                horizontalalignment='center',
                color='white' if cm[i,j] > thres else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicated label')

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(10))
# Display some error results

errors = (y_pred_classes - y_true) != 0

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title('Pred: {}, true: {}'.format(pred_errors[error], obs_errors[error]))
            n += 1

y_pred_errors_prob = np.max(y_pred_errors, axis=1)
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors
sorted_delta_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_delta_errors[-6:]

display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)
results = model.predict(X_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label', dtype=np.int)
submission = pd.Series(range(1, X_test.shape[0]+1), name='ImageId', dtype=np.int)
submission = pd.concat([submission, results], axis=1)
submission.head()
submission.to_csv('submission.csv', index=False)
