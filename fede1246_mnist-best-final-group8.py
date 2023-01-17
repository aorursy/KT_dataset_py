# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
image_train = pd.read_csv('../input/fashion-mnist_train.csv')

image_test = pd.read_csv('../input/fashion-mnist_test.csv')
print('Shape Train:', image_train.shape)

print('Shape Test:', image_test.shape)
image_train.head(5)
image_test.head(5)
#create model

model = Sequential()



#let's add some layers baby!

model.add((Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28,28,1))))

model.add((Conv2D(32, kernel_size = 3, activation = 'relu')))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5)



print(history.history.keys())



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(x_test)
from sklearn import metrics

import seaborn as sns



cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

cm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cm)

fig_corr, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g', cmap='Blues', ax=ax)

plt.show()
#create model

model = Sequential()



#let's add some layers baby!

model.add((Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28,28,1))))

model.add(Dropout(0.25))

model.add((Conv2D(32, kernel_size = 3, activation = 'relu')))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history1 = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5)



print(history1.history.keys())



plt.plot(history1.history['acc'])

plt.plot(history1.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history1.history['loss'])

plt.plot(history1.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(x_test)
from sklearn import metrics

import seaborn as sns



cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

cm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cm)

fig_corr, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g', cmap='Blues', ax=ax)

plt.show()
#accuracy

plt.figure(figsize=(10,10))



plt.plot(history1.history['acc'], color = 'darkorange')

plt.plot(history1.history['val_acc'], color = 'orange' )

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')



plt.plot(history.history['acc'], color = 'darkred' )

plt.plot(history.history['val_acc'], color = 'brown' )

plt.legend(['Train D = 0.25','Test D = 0.25', 'Train D = 0', 'Test D = 0'], loc='upper left')



plt.show()



#loss

plt.figure(figsize=(10,10))

plt.plot(history1.history['loss'], color = 'darkorange')

plt.plot(history1.history['val_loss'], color = 'orange')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')



plt.plot(history.history['loss'], color = 'darkred')

plt.plot(history.history['val_loss'], color = 'brown')



plt.legend(['Train D = 0.25','Test D = 0.25', 'Train D = 0', 'Test D = 0'], loc='upper left')



plt.show()



#create model

model = Sequential()



#let's add some layers baby!

model.add((Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28,28,1))))

model.add(Dropout(0.4))

model.add((Conv2D(32, kernel_size = 3, activation = 'relu')))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history2 = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5)

#accuracy

plt.figure(figsize=(10,10))

plt.plot(history2.history['acc'], color = 'darkslategrey')

plt.plot(history2.history['val_acc'], color = 'teal' )



plt.plot(history1.history['acc'], color = 'darkorange')

plt.plot(history1.history['val_acc'], color = 'orange' )

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')



plt.plot(history.history['acc'], color = 'darkred' )

plt.plot(history.history['val_acc'], color = 'brown' )

plt.legend(['Train D = 0.4','Test D = 0.4','Train D = 0.25','Test D = 0.25', 'Train D = 0', 'Test D = 0'], loc='upper left')



plt.show()



#loss

plt.figure(figsize=(10,10))

plt.plot(history2.history['acc'], color = 'darkslategrey')

plt.plot(history2.history['val_acc'], color = 'teal' )



plt.plot(history1.history['loss'], color = 'darkorange')

plt.plot(history1.history['val_loss'], color = 'orange')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')



plt.plot(history.history['loss'], color = 'darkred')

plt.plot(history.history['val_loss'], color = 'brown')



plt.legend(['Train D = 0.4','Test D = 0.4','Train D = 0.25','Test D = 0.25', 'Train D = 0', 'Test D = 0'], loc='upper left')



plt.show()


