# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



import matplotlib.pyplot as plt #for plotting

from collections import Counter

from sklearn.metrics import confusion_matrix

import itertools

import seaborn as sns

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls /kaggle/input/Kannada-MNIST/
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
y = train['label']

X = train.drop(['label'],axis=1)

X_train = X.to_numpy()

X_train = np.reshape(X_train,(60000,28,28,1))

y_train = y.to_numpy()

X_train, X_aval, y_train, y_aval = train_test_split(X_train, y_train, test_size=0.4)
x_train = X_train.astype('float32')

x_aval = X_aval.astype('float32')

x_train /= 255

x_aval /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_aval.shape[0], 'aval samples')
batch_size = 32

num_classes = 10

epochs = 12# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_aval = keras.utils.to_categorical(y_aval, num_classes)
input_shape = (28, 28, 1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_aval, y_aval))
score = model.evaluate(x_aval, y_aval, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
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
# Predict the values from the validation dataset

Y_pred = model.predict(x_aval)



# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_aval, axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))

X_test = test.drop(['id'],axis=1)

X_test = X_test.to_numpy()

X_test = np.reshape(X_test,(X_test.shape[0],28,28,1))

x_test = X_test.astype('float32')

x_test /= 255

# Predict the values from the validation dataset

Y_pred_test = model.predict(x_test)



# Convert predictions classes to one hot vectors 

Y_pred_classes_test = np.argmax(Y_pred_test, axis = 1) 

a = np.arange(0,5000,1).reshape(-1,1)

b = Y_pred_classes_test.reshape(-1,1)

resultado = np.concatenate((a,b),axis=1)

df_resultado = pd.DataFrame(data = resultado, columns=['id','label'])

df_resultado.head()

df_resultado.to_csv('submission.csv', index=False)