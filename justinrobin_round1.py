#loading the imgs to the environment

!tar xzvf /kaggle/input/cifar10-python/cifar-10-python.tar.gz

def load_data():

    """Loads CIFAR10 dataset.

    Returns:

      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    """

    import os

    import sys

    from six.moves import cPickle

    

    def load_batch(fpath):

        with open(fpath, 'rb') as f:

            d = cPickle.load(f, encoding='bytes')  

        data = d[b'data']

        labels = d[b'labels']

        data = data.reshape(data.shape[0], 3, 32, 32)

        return data, labels

    

    path = 'cifar-10-batches-py'

    num_train_samples = 50000



    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')

    y_train = np.empty((num_train_samples,), dtype='uint8')



    for i in range(1, 6):

        fpath = os.path.join(path, 'data_batch_' + str(i))

        (x_train[(i - 1) * 10000:i * 10000, :, :, :],

         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    

    x_test, y_test = load_batch(os.path.join(path, 'test_batch'))



    y_train = np.reshape(y_train, (len(y_train), 1))

    y_test = np.reshape(y_test, (len(y_test), 1))



    x_train = x_train.transpose(0, 2, 3, 1)

    x_test = x_test.transpose(0, 2, 3, 1)



    return (x_train, y_train), (x_test, y_test)
from PIL import Image



from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Dropout

from keras.layers import Convolution2D, MaxPooling2D

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam
import numpy as np

#I'm not resizing the img as it's all ready small

(x_train, y_train), (x_test, y_test) = load_data()

print(np.unique(y_train))

nb_classes = len(np.unique(y_train))

print('the num of classes',nb_classes)
#kinda Noramalizing the img

x_train = x_train.astype('float32')/255.

x_test = x_test.astype('float32')/255.



y_train = to_categorical(y_train, nb_classes)

y_test = to_categorical(y_test, nb_classes)
#we have 50000 imgs with the size of 32*32 and are RBG in nature

print(x_train.shape)

print(y_train.shape)

#we have 10000 imgs with the size of 32*32 and are RBG in nature

print(x_test.shape)

print(y_test.shape)
#Sample, lets polt a image and see what it looks like

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,10))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(x_train[i])

    plt.axis('off')
model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same',

                        input_shape=(32,32,3), 

                        activation='relu'))

model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))



adam = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=adam)

model.summary()
base_model= model.fit(x_train, y_train, batch_size=60,epochs=10, verbose=1, validation_data=(x_test, y_test))
from pylab import rcParams

rcParams['figure.figsize'] = 10, 4

plt.plot(base_model.history['accuracy'])

plt.plot(base_model.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(base_model.history['loss'])

plt.plot(base_model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_2 = Sequential()



model_2.add(Conv2D(32,(3,3),activation = "relu", input_shape=(32,32,3)))

model_2.add(Conv2D(32,(3,3),activation = "relu"))

model_2.add(MaxPooling2D(pool_size=(3, 3)))

model_2.add(Dropout(0.5))

model_2.add(Dense(64, activation='relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Dropout(0.5))



model_2.add(Flatten())

model_2.add(Dense(64, activation='relu'))

model_2.add(Dense(nb_classes, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002), metrics=["accuracy"])

#Learning rate(lr) can a be a game changer but as

#we all know the smaller the better for the GD and to find the Globel min.

model_2.summary()
cnn_2 = model_2.fit(x_train, y_train, batch_size=100, epochs=100, verbose=1,validation_data=(x_test, y_test),

                    callbacks=[early_stopping])
rcParams['figure.figsize'] = 10, 4

plt.plot(cnn_2.history['accuracy'])

plt.plot(cnn_2.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_2.history['loss'])

plt.plot(cnn_2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#Knowing that this a models ability to learn is quit good so

#save em load em, rum em!

from keras.models import load_model

model_2.save_weights('Cnn_weights.h5')
#Know that my model is outsmarting the test fellow, just gen more image to both the side, hopping to leave out the plane on both end.

from keras.preprocessing.image import ImageDataGenerator

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

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)





datagen.fit(x_test)
model_2 = Sequential()



model_2.add(Conv2D(32,(3,3),activation = "relu", input_shape=(32,32,3)))

model_2.add(Conv2D(32,(3,3),activation = "relu"))

model_2.add(MaxPooling2D(pool_size=(2, 2)))



model_2.add(Dropout(0.5))

model_2.add(Conv2D(64,(3,3),activation = "relu"))



model_2.add(Flatten())

model_2.add(Dense(64, activation='relu'))

model_2.add(Dense(nb_classes, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])

model_2.summary()
# Transfer-learning

model_2.load_weights('Cnn_weights.h5', by_name=True)





cnn_2 = model_2.fit(x_train, y_train, batch_size=60, epochs=100, verbose=1,validation_data=(x_test, y_test),

                    callbacks=[early_stopping])
model_2.save_weights('Cnn_2_weights.h5')
rcParams['figure.figsize'] = 10, 4

plt.plot(cnn_2.history['accuracy'])

plt.plot(cnn_2.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_2.history['loss'])

plt.plot(cnn_2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#Almost close to VGGs architecture, atlest this architecture follows the flow of the how the architecture

#was made.

model_3 = Sequential()



model_3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))

model_3.add(Conv2D(32, (3, 3), activation='relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2)))



model_3.add(Conv2D(16, (3, 3), activation='relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2)))



model_3.add(Conv2D(8, (3, 3), activation='relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2)))



model_3.add(Flatten())

model_2.add(Dense(32, activation='relu'))

model_3.add(Dense(nb_classes, activation='sigmoid'))





model_3.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.002), metrics=["accuracy"])

model_3.summary()
model_3.load_weights('Cnn_2_weights.h5', by_name=True)



cnn_3 = model_3.fit(x_train, y_train, batch_size=60, epochs=70, verbose=1,validation_data=(x_test, y_test),

                    callbacks=[early_stopping])
rcParams['figure.figsize'] = 10, 4

plt.plot(cnn_3.history['accuracy'])

plt.plot(cnn_3.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(cnn_2.history['loss'])

plt.plot(cnn_2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model_3.save_weights('Cnn_3=cnn2+1_weights.h5')