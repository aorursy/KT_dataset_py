import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

subm = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
train.head()
X_train , X_test , y_train , y_test = train_test_split(train.iloc[: , 1:].values , train.iloc[:,0].values , test_size = 0.1)
fig, ax = plt.subplots(2, 1, figsize=(12,6))

ax[0].plot(X_train[5])

ax[0].set_title('784x1 data')

ax[1].imshow(X_train[0].reshape(28,28), cmap='gray')

ax[1].set_title('28x28 data')
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
X_train = X_train.reshape(-1 , 28 , 28 , 1)/255

X_test = X_test.reshape(-1 , 28 , 28 , 1)/255
model = Sequential()

model.add(Conv2D(filters = 16 , kernel_size= (3,3) , activation= 'relu' , input_shape = (28 , 28 ,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters= 16 , kernel_size= (3,3) , activation= 'relu' ))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = model.fit_generator(datagen.flow(X_train , y_train , batch_size=16) ,

                          steps_per_epoch = 500 ,

                           epochs = 20,

                           verbose =2,

                           validation_data=(X_test[:400,:], y_test[:400,:]),

                           callbacks=[annealer]

                          )
final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)

print('loss =' , final_loss , '\naccuracy' , final_acc)
test.shape
test.head()
test = test.iloc[:,1:].values 

test = test.reshape(-1 , 28 , 28 , 1)/255



pre = model.predict(test, batch_size=64)
pre = np.argmax(pre , axis = 1)
subm.head()
pre
subm['label'] = pre
subm.to_csv('submit.csv' , index= False)