# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models,layers,optimizers

from keras.layers import BatchNormalization,Dropout

from keras import models

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.datasets import mnist



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

train_len = len(train)

test = pd.read_csv("../input/test.csv")

#data.head(10)

y = train.label

train.drop(labels = 'label',axis = 1,inplace = True)

data = pd.concat([train,test])
new_data = data.values

new_data = new_data.reshape((70000,28,28,1))

new_data = new_data.astype('float32') / 255

new_data.shape

train_data = new_data[:train_len]

test_data = new_data[train_len:]

train_labels = to_categorical(y)

train_data.shape
model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.25))  

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(Dropout(0.5))  

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',

            loss='categorical_crossentropy',

            metrics=['accuracy'])

model.summary()


train_X, val_X,train_y,val_y = train_test_split(train_data,train_labels, test_size=0.20, random_state=42)

total_train = train_X.shape[0]

print(total_train)

total_val = val_X.shape[0]

batch_size=40

epochs = 30

print(train_X.shape)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
#history=model.fit(train_X, train_y,validation_data=(val_X,val_y), epochs=epochs, batch_size=batch_size)  (accuracy = 0.99 without augmentation)

#augmenting

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

    #rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=False,)



train_datagen.fit(train_X)
history = model.fit_generator(train_datagen.flow(train_X,train_y, batch_size=batch_size),

                              epochs = epochs, validation_data = (val_X,val_y),

                              verbose = 2, steps_per_epoch=train_X.shape[0] // batch_size

                              , callbacks=callbacks)
model.save('digit_recognizer.h5')



acc=history.history['acc']

val_acc = history.history['val_acc']

loss=history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)



plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model.fit(train_data,train_labels,epochs=21,batch_size=batch_size)
from sklearn.preprocessing import LabelEncoder

test_preds = model.predict(test_data)

predictions = [i.tolist().index(max(i)) for i in test_preds]
output = pd.DataFrame({'ImageId': range(1,len(test_data)+1),

                      'Label': predictions})

output.to_csv('submission.csv', index=False)