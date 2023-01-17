import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm



from keras.models import Sequential

from keras.layers import Dense, Conv2DTranspose,Conv2D, MaxPooling2D

from keras.layers import Dropout, Flatten

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
g = plt.imshow(X_train[10][:,:,0])


def baseline_model():

    model = Sequential()

    model.add(Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(64, 3, strides=1, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(128, 3, strides=1, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(128, 3, strides=1, padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(1024, 

                activation='relu'))

    model.add(Dense(1024, 

                activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, 

                activation='softmax'))

    return model
model = baseline_model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)



datagen.fit(X_train)
history = model.fit_generator(

                            datagen.flow(X_train,Y_train, batch_size=50),

                            epochs = 8, 

                            validation_data = (X_val,Y_val),

                            verbose = 1,

                            steps_per_epoch=X_train.shape[0] // 50

)
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)