import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

%matplotlib inline
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

print(train.shape)

train.head()
sns.countplot(train['label'])
x_train = (train.iloc[:,1:].values).astype('float32') # all pixels

y_train = train.iloc[:,0].values.astype('int32') # all labels

x_test = test.values.astype('float32') # all pixels
%matplotlib inline

# preview the images first

plt.figure(figsize=(12,10))

x, y = 10, 4

for i in range(40):  

    plt.subplot(y, x, i+1)

    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')

plt.show()
x_train = x_train/255.0

x_test = x_test/255.0
print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
X_train = x_train.reshape(x_train.shape[0], 28, 28,1)

X_test = x_test.reshape(x_test.shape[0], 28, 28,1)
y_train = keras.utils.to_categorical(y_train, 10)



X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)
batch_size = 64

epochs = 20

input_shape = (28, 28, 1)
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), 

                    epochs = epochs, validation_data = (X_val,Y_val),

                    verbose = 1, 

                    steps_per_epoch = X_train.shape[0] // batch_size

                   )
predictions = model.predict(X_test)

results = np.argmax(predictions, axis = 1)
submissions = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submissions['Label'] = results

submissions.to_csv('submission.csv', index = False)