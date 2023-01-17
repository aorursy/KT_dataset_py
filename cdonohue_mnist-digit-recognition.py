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
import numpy as np

mnist_file = "../input/train.csv"

data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')

print(data.shape)

img_rows, img_cols = 28, 28

num_classes = 10
import keras

y = data[:, 0]

out_y = keras.utils.to_categorical(y, num_classes)

out_y.shape
x = data[:,1:]

x.shape
num_images = data.shape[0]

out_x = x.reshape(num_images, img_rows, img_cols, 1)

out_x = out_x / 255

out_x.shape
x = out_x

y = out_y
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



mnist_model = Sequential()
mnist_model.add(Conv2D(12, kernel_size=3, activation="relu", input_shape=(img_rows,img_cols,1)))

mnist_model.add(Conv2D(20, activation='relu', kernel_size=3, strides=2))

mnist_model.add(Conv2D(20, activation='relu', kernel_size=3))

mnist_model.add(Flatten())

mnist_model.add(Dense(100, activation='relu'))

mnist_model.add(Dense(num_classes, activation='softmax'))

mnist_model.summary()
mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=3)]

mnist_model.fit(x, y, batch_size=100, epochs = 30, validation_split = 0.2, callbacks = callbacks_list, verbose=1)
result = mnist_model.predict(x, verbose=1)
import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels





def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
actual = data[:, 0].astype(int)
predict = result.argmax(axis=1)
classes = np.unique(data[:,0])
# Plot non-normalized confusion matrix

plot_confusion_matrix(actual, predict, classes=classes,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(actual, predict, classes=classes, normalize=True,

                      title='Normalized confusion matrix')
mnist_file = "../input/test.csv"

data = np.loadtxt(mnist_file, skiprows=1, delimiter=',')

print(data.shape)

img_rows, img_cols = 28, 28

num_classes = 10
y = data[:, 0]

out_y = keras.utils.to_categorical(y, num_classes)

print(out_y.shape)
x = data[:,:]

print(x.shape)

num_images = data.shape[0]

out_x = x.reshape(num_images, img_rows, img_cols, 1)

out_x = out_x / 255
x = out_x

y = out_y
result = mnist_model.predict(x, verbose=1)
f = open('submission.csv', 'w')

f.write('ImageId,Label\n')

for index in range(0,num_images):

    predict = np.argmax(result[index])

    f.write('%i,%i\n' % (index+1, predict))

f.close()