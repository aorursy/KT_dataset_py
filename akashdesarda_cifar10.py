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
from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers



import numpy as np



seed = 7

np.random.seed(seed)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



mean = np.mean(x_train,axis=(0,1,2,3))

std = np.std(x_train,axis=(0,1,2,3))

x_train = (x_train - mean)/(std + 1e-7)

x_test = (x_test - mean)/(std + 1e-7)
num_class = 10

y_train = to_categorical(y_train, )

y_test = to_categorical(y_test)
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = (32, 32, 3),padding='same', activation = 'relu'))

model.add(Dropout(0.2))



model.add(Conv2D(32,(3,3),padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add((Dropout(0.2)))

model.add(Dense(no_class, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=96)
model.evaluate(x_test, y_test)
model.evaluate(x_test, y_test)
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = (32, 32, 3),padding='same', activation = 'relu'))

model.add(Dropout(0.2))



model.add(Conv2D(32,(3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D())



model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))

model.add(Dropout(0.20))



model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D())



model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))

model.add(Dropout(0.20))



model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dropout(0.20))



model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.20))



model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.20))



model.add(Dense(no_class, activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test),

          epochs=100, batch_size=120)
model.evaluate(x_test, y_test)
test = np.load('../input/features.npy')
test.shape
Y_pred = model.predict_classes(test)
Y_pred.shape
(y_train[1000])
submission = pd.DataFrame(columns=['id','label'])

a = np.array(range(1,300001))
submission['id'] = a

submission['label'] = Y_pred
submission[(submission['label']==0)] = 'airplane'

submission[(submission['label']==1)] = 'automobile'

submission[(submission['label']==2)] = 'bird'

submission[(submission['label']==3)] = 'cat'

submission[(submission['label']==4)] = 'deer'

submission[(submission['label']==5)] = 'dog'

submission[(submission['label']==6)] = 'frog'

submission[(submission['label']==7)] = 'horse'

submission[(submission['label']==8)] = 'ship'

submission[(submission['label']==9)] = 'truck'
a = np.array(range(1,300001))
submission.to_csv('submission_30-6.csv', index=None)
submission['id'] = a

submission.head()
model = Sequential()



model.add(Conv2D(32,(3,3), input_shape = (32, 32, 3),padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))

model.add(BatchNormalization())

model.add(Conv2D(32,(3,3), padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3), padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3), padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.3))



model.add(Conv2D(128,(3,3), padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3), padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(0.0001)))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(num_class, activation='softmax'))



model.summary()
# Data Augmentation



datagen = ImageDataGenerator(rotation_range=15, 

                             width_shift_range=0.1,

                             height_shift_range=0.1,

                             horizontal_flip=True)

datagen.fit(x_train)
model.compile(loss='categorical_crossentropy', 

              optimizer='adam',

              metrics=['accuracy'])



model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),

                    steps_per_epoch=x_train.shape[0] // 64,epochs=50,

                    validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)
y_pred = model.predict(test)
pred = np.argmax(y_pred, axis=1)
np.greater(np.argmax(y_pred[0]), 0.55)
submission = pd.DataFrame(columns=['id','label'])

a = np.array(range(1,300001))
submission['id'] = a

submission['label'] = pred
submission.head()
submission[(submission['label']==0)] = 'airplane'

submission[(submission['label']==1)] = 'automobile'

submission[(submission['label']==2)] = 'bird'

submission[(submission['label']==3)] = 'cat'

submission[(submission['label']==4)] = 'deer'

submission[(submission['label']==5)] = 'dog'

submission[(submission['label']==6)] = 'frog'

submission[(submission['label']==7)] = 'horse'

submission[(submission['label']==8)] = 'ship'

submission[(submission['label']==9)] = 'truck'

submission.head()
submission['id'] = a
submission.head()
submission.to_csv('submission_02-07.csv', index=None)