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

import numpy as np

import matplotlib.pyplot as plt

from keras import Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

from keras.losses import categorical_crossentropy

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
TRAIN_DF = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

TEST_DF = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

TRAIN_DF.head()
X = TRAIN_DF.drop(['label'],axis=1).values

y = TRAIN_DF.label.values
X = X.reshape(-1,28,28,1)

y = to_categorical(y,num_classes=10)
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2)
TRAIN_GEN = ImageDataGenerator(shear_range=0.2,

                               zoom_range=0.2,

                               horizontal_flip=False,

                               height_shift_range=0.1,

                               width_shift_range=0.1,

                               rotation_range=10,

                               rescale=1.0/255.0)

VALID_GEN = ImageDataGenerator(rescale=1.0/255.0)
TRAIN_ITER = TRAIN_GEN.flow(X_train,y_train)

VALID_ITER = VALID_GEN.flow(X_valid,y_valid)
first_batch = TRAIN_ITER.next()

_,ax = plt.subplots(1,10,figsize=(16,5))

i = 0

for image,label in zip(first_batch[0][:10],first_batch[1][:10]):

    ax[i].imshow(image.reshape(28,28),cmap='binary')

    ax[i].set_title(str(np.argmax(label)))

    i+=1
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',

             input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
early_stopping = EarlyStopping(restore_best_weights=True,patience=5)

lr_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                patience=3, 

                                factor=0.5,

                                min_lr=0.00001)
history = model.fit_generator(TRAIN_ITER,

                    steps_per_epoch=TRAIN_ITER.n//TRAIN_ITER.batch_size,

                    epochs=50,

                    validation_data=VALID_ITER,

                    validation_steps=VALID_ITER.n//VALID_ITER.batch_size,

                    shuffle=True,

                    callbacks=[early_stopping,lr_reduction])
_,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(history.history['acc'],label='training_accuracy')

ax[0].plot(history.history['val_acc'],label='validation_accuracy')

ax[0].legend()

ax[0].set_xlabel('Epochs')

ax[0].set_ylabel('Accuracy')

ax[0].set_title("Accuracy")



ax[1].plot(history.history['loss'],label='training_loss')

ax[1].plot(history.history['val_loss'],label='validation_loss')

ax[1].legend()

ax[1].set_xlabel('Epochs')

ax[1].set_ylabel('Loss')

ax[1].set_title("Loss")
Test_Data = TEST_DF.drop(['id'],axis=1).values

Test_Data = Test_Data.reshape(-1,28,28,1)

Test_Data = Test_Data/255

predictions = model.predict(Test_Data)
Submission_DF = pd.DataFrame()

ImageId = []

Label = []

for IMG_ID,prediction in enumerate(predictions):

    ImageId.append(IMG_ID)

    Label.append(np.argmax(prediction))

Submission_DF['id'] = ImageId

Submission_DF['label'] = Label

Submission_DF.to_csv('submission.csv',index=False)