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
import pandas as pd

data = pd.read_csv('../input/Kannada-MNIST/train.csv')

data.append(pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv'))
data.hist(column='label')
import matplotlib.pyplot as plt



image = data.drop('label', axis=1)

image = image.iloc[0].values.reshape(28,28)

plt.subplot(131)

plt.title(data.iloc[0]['label'])

plt.imshow(image)



image = data.drop('label', axis=1)

image = image.iloc[1].values.reshape(28,28)

plt.subplot(132)

plt.title(data.iloc[1]['label'])

plt.imshow(image)



image = data.drop('label', axis=1)

image = image.iloc[2].values.reshape(28,28)

plt.subplot(133)

plt.title(data.iloc[2]['label'])

plt.imshow(image)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.33)
X_train
X_test
y_train
y_test
X_train = X_train.values.reshape(len(X_train), 28,28,1)
X_test = X_test.values.reshape(len(X_test), 28,28,1)
y_train = pd.get_dummies(y_train)

y_train
y_test = pd.get_dummies(y_test)

y_test
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Convolution1D, Dropout, MaxPooling2D, Input, Activation

from keras.models import Model

from keras.optimizers import SGD

from tensorflow.keras.layers import Dense, Flatten, Conv2D



img_rows = 28

img_cols = 28

num_classes = 10



model = Sequential()

model.add(Conv2D(12, kernel_size=3, activation="relu", input_shape=(img_rows,img_cols,1)))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(20, activation='relu', kernel_size=3, strides=2))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, activation='relu', kernel_size=3))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)]

history = model.fit(X_train, y_train, batch_size=32, epochs = 100, validation_data=(X_test, y_test), callbacks = callbacks_list, verbose=1, shuffle=True)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')



plt.legend(['train', 'val'], loc='upper left')

plt.show()
print(model.metrics_names)

model.evaluate(X_test, y_test, verbose=1)
y_pred = pd.DataFrame(model.predict_classes(X_test, verbose=1))
y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix((y_test.iloc[:, 1:] == 1).idxmax(1), y_pred)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

#ax.set_xticklabels([''] + labels)

#ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
data = pd.read_csv('../input/Kannada-MNIST/test.csv')

data
pred_y = model.predict_classes(data.drop('id', axis=1).values.reshape(len(data),28,28,1), verbose=1)
pred_y = pd.DataFrame(pred_y, columns=['label'])
pred_y['id'] = data['id']
pred_y.to_csv('submission.csv', index=False, columns=['id', 'label'])
#file = open('submission.csv', 'r') 

#print(file.read())