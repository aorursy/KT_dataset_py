import numpy

from keras import backend as K

from keras.datasets import mnist

from keras.utils import np_utils
from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

import pandas as pd
K.set_image_data_format('channels_last')

numpy.random.seed(0)
X = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



y = X["label"]

X.drop(["label"], inplace = True, axis = 1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2 , random_state=42)
import matplotlib.pyplot as plt

print("the number of training examples = %i" % X_train.shape[0])

print("the number of classes = %i" % len(numpy.unique(y_train)))

print("Dimention of images = {:d} x {:d}  ".format(X_train[1].shape[0],X_train[1].shape[1])  )



#This line will allow us to know the number of occurrences of each specific class in the data

unique, count= numpy.unique(y_train, return_counts=True)

print("The number of occuranc of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )

 

images_and_labels = list(zip(X_train,  y_train))

for index, (image, label) in enumerate(images_and_labels[:12]):

    plt.subplot(5, 4, index + 1)

    plt.axis('off')

    plt.imshow(image.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title('label: %i' % label )
model = Sequential()



from keras.layers import Dropout



model.add(Conv2D(20, kernel_size=11, padding="valid", input_shape=(28, 28, 1), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(40, kernel_size=3, padding="same", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Conv2D(50, kernel_size=3, padding="valid", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
from keras.layers.core import Activation



model.add(Flatten())

# model.add(Dense(units=1000, activation='relu'  ))

model.add(Dropout(0.2))



model.add(Dense(10))

model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train = np_utils.to_categorical(y_train).astype('int32')

y_test = np_utils.to_categorical(y_test)
from tensorflow import keras



callbacks = [

    keras.callbacks.EarlyStopping(

        # Stop training when `val_loss` is no longer improving

        monitor='val_loss',

        # "no longer improving" being defined as "no better than 1e-2 less"

        min_delta=1e-1,

        # "no longer improving" being further defined as "for at least 2 epochs"

        patience=100,

        verbose=1)

]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=5,

    fill_mode='nearest',

    validation_split = 0.2

    )



datagen.fit(X_train)



train_generator = datagen.flow(X_train, y_train, batch_size=60, subset='training')



validation_generator = datagen.flow(X_train, y_train, batch_size=60, subset='validation')





# # fits the model on batches with real-time data augmentation:

history = model.fit_generator(generator=train_generator,

                    validation_data=validation_generator,

                    callbacks = callbacks,

                    use_multiprocessing=True,

                    steps_per_epoch = len(train_generator) / 60,

                    validation_steps = len(validation_generator) / 60,

                    epochs = 400,

                    workers=20)