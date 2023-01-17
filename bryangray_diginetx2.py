# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.callbacks import Callback

tf.__version__
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

sample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train.head()
train.describe()
train['label'].value_counts()
Y = train["label"].values

X = train.drop(labels = ["label"],axis = 1).values 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)

X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

# Final testing

X_final = test.values.reshape(-1,28,28,1)
g = plt.imshow(X_train[0][:,:,0])
plt.imshow(X_test[3][:,:,0])
# Defining a sequential model

model = Sequential()

model.add(Conv2D(16, (3,3), padding='same', activation='relu', input_shape = (28, 28, 1)))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))



# Using RMSprop optimizer, sparse categorical cross entropy loss

model.compile(

    optimizer=RMSprop(lr=0.0001), 

    loss="sparse_categorical_crossentropy", 

    metrics=["acc"]

)
model.summary()
train_datagen = ImageDataGenerator(

    rescale=1./255,

#     rotation_range=40,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     shear_range=0.2,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     fill_mode='nearest'

)

test_datagen = ImageDataGenerator(

    rescale=1./255

)
# Hyperparameters

epochs = 50

max_accuracy = 0.9999999

batch_size = 64

steps_per_epoch=len(X_train)//64

validation_steps=len(X_test)//64
class myCallback(Callback):

    def on_epoch_end(self, epoch, logs={}):

        if logs.get("acc")>=max_accuracy:

            print("\nReached {}% accuracy!".format(max_accuracy*100))

            self.model.stop_training = True 

            

callback = myCallback()
history = model.fit_generator(

    train_datagen.flow(X_train, Y_train, batch_size=batch_size),

    epochs = epochs,

    steps_per_epoch=steps_per_epoch,

    callbacks=[callback],

    validation_data=test_datagen.flow(X_test, Y_test),

    validation_steps=validation_steps,

    verbose=1

)
# Plot history

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

results = model.predict(X_final)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)





results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist.csv",index=False)
submission.describe()
submission['Label'].value_counts()