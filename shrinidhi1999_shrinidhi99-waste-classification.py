from keras.models import Sequential

from keras.layers import Conv2D, Activation,Dropout

from keras.models import Model,load_model

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras.layers.core import Flatten, Dense

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.engine.topology import Layer

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_files

import itertools

import numpy as np

import pandas as pd

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import itertools



train_dir = '/kaggle/input/waste-classification-data/dataset/DATASET/TRAIN'

test_dir = '/kaggle/input/waste-classification-data/dataset/DATASET/TEST'



def load_dataset(path):

    data = load_files(path) #load all files from the path

    files = np.array(data['filenames']) #get the file  

    targets = np.array(data['target'])#get the the classification labels as integer index

    target_labels = np.array(data['target_names'])#get the the classification labels 

    return files,targets,target_labels

    

x_train, y_train,target_labels = load_dataset(train_dir)

x_test, y_test,_ = load_dataset(test_dir)



print('Training set size : ' , x_train.shape[0])

print('Testing set size : ', x_test.shape[0])



x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 1)



print ("x_train shape: " + str(x_train.shape))

print ("y_train shape: " + str(y_train.shape))

print ("x_validate shape: " + str(x_validate.shape))

print ("y_validate shape: " + str(y_validate.shape))

print ("x_test shape: " + str(x_test.shape))

print ("y_test shape: " + str(y_test.shape))



def convert_image_to_array(files):

    width, height, channels = 100, 100, 3

    images_as_array = np.empty((files.shape[0], width, height, channels), dtype=np.uint8) #define train and test data shape

    for idx,file in enumerate(files):

        img = cv2.imread(file) 

        res = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC) #As images have different size, resizing all images to have same shape of image array

        images_as_array[idx] = res

    return images_as_array



x_train = np.array(convert_image_to_array(x_train))

print('Training set shape : ',x_train.shape)



x_valid = np.array(convert_image_to_array(x_validate))

print('Validation set shape : ',x_valid.shape)



x_test = np.array(convert_image_to_array(x_test))

print('Test set shape : ',x_test.shape)



x_train = x_train.astype('float32')/255

x_valid = x_valid.astype('float32')/255

x_test = x_test.astype('float32')/255

y_train = y_train.reshape(y_train.shape[0],1)

y_test = y_test.reshape(y_test.shape[0],1)

y_validate = y_validate.reshape(y_validate.shape[0],1)
!ls
def CNN_model():

    dense_layers = [0, 1, 2]

    layer_sizes = [64, 128]

    conv_layers = [1, 2, 3]

    for dense_layer in dense_layers:

        for layer_size in layer_sizes:

            for conv_layer in conv_layers:

                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(

                    conv_layer, layer_size, dense_layer, int(time.time()))

                print(NAME)



                model = Sequential()



                model.add(Conv2D(layer_size, (3, 3), input_shape=(100,100,3)))

                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))



                for l in range(conv_layer-1):

                    model.add(Conv2D(layer_size, (3, 3)))

                    model.add(Activation('relu'))

                    model.add(MaxPooling2D(pool_size=(2, 2)))



                model.add(Flatten())



                for _ in range(dense_layer):

                    model.add(Dense(layer_size))

                    model.add(Activation('relu'))

                    model.add(Dropout(0.5))



                model.add(Dense(1))

                model.add(Activation('sigmoid'))

    #             print('came here')



                model.summary()

    return model
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0, # Randomly zoom image 

        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

print(x_train.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf

import pickle

import time

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint





# pickle_in = open("X.pickle", "rb")

# X = pickle.load(pickle_in)



# pickle_in = open("y.pickle", "rb")

# y = pickle.load(pickle_in)



# X = X/255.0

IMG_SIZE = 100

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)



# y = np.array(y)



opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model = CNN_model() # without data augmentation



checkpoint = ModelCheckpoint('128x3-CNN-no-aug.hdf5',  # model filename

                             monitor='val_loss', # quantity to monitor

                             verbose=0, # verbosity - 0 or 1

                             save_best_only= True, # The latest best model will not be overwritten

                             mode='auto') # The decision to overwrite model is made 

                                          # automatically depending on the quantity to monitor



model.compile(loss='binary_crossentropy',

                optimizer=opt,

                metrics=['accuracy'])



earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement

                          min_delta = 0, #Abs value and is the min change required before we stop

                          patience = 15, #Number of epochs we wait before stopping 

                          verbose = 1,

                          restore_best_weights = True) #keeps the best weigths once stopped



ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)

callbacks = [earlystop, checkpoint, ReduceLR]



# model_details = model.fit(X, y,

#                     batch_size = 32,

#                     epochs = 12, # number of iterations

#                     validation_split=0.2,

#                     callbacks=[checkpoint],

#                     verbose=1)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size= 32), epochs = 20, verbose=1,callbacks = callbacks,validation_data=(x_valid,y_validate))

# model.fit(X, y,

#             batch_size=32,

#             epochs=10,

#             validation_split=0.2)



model.save('128x3-CNN-no-aug.hdf5')
import pickle



pickle_out = open("Trained_cnn_history.pickle","wb")

pickle.dump(history.history, pickle_out)

pickle_out.close()



pickle_in = open("Trained_cnn_history.pickle","rb")

saved_history = pickle.load(pickle_in)

print(saved_history)



import pickle

import matplotlib.pyplot as plt

import tensorflow as tf



model = tf.keras.models.load_model("128x3-CNN-no-aug.hdf5")



score_train = model.evaluate(x_train, y_train, verbose=0)

print('\n\nTrain Loss: ', score_train[0])

print('Train Accuracy: ', score_train[1])



score = model.evaluate(x_test,y_test,verbose=0)

print('\nTest Loss :',score[0])

print('Test Accuracy :',score[1])



#get the predictions for the test data

predicted_classes = model.predict_classes(x_test)



confusion_mtx = confusion_matrix(y_test, predicted_classes) 



plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('confusion_matrix')

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ['R','O'], rotation=90)

plt.yticks(tick_marks, ['R','O'])

#Following is to mention the predicated numbers in the plot and highligh the numbers the most predicted number for particular label

thresh = confusion_mtx.max() / 2.

for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):

    plt.text(j, i, confusion_mtx[i, j],

    horizontalalignment="center",

    color="white" if confusion_mtx[i, j] > thresh else "black")



plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf

import pickle

import time

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint

import keras





# pickle_in = open("X.pickle", "rb")

# X = pickle.load(pickle_in)



# pickle_in = open("y.pickle", "rb")

# y = pickle.load(pickle_in)



# X = X/255.0

IMG_SIZE = 100

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)



# y = np.array(y)



opt = tf.keras.optimizers.Adam(learning_rate=0.001)



dense_layers = [2]

layer_sizes = [128]

conv_layers = [3]

for dense_layer in dense_layers:

    for layer_size in layer_sizes:

        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(

                conv_layer, layer_size, dense_layer, int(time.time()))

            print(NAME)



            model = Sequential()



            model.add(Conv2D(layer_size, (3, 3), input_shape=(100,100,3)))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))



            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3, 3)))

                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))



            model.add(Flatten())



            for _ in range(dense_layer):

                model.add(Dense(layer_size))

                model.add(Activation('relu'))

                model.add(Dropout(0.5))



            model.add(Dense(1))

            model.add(Activation('sigmoid'))

#             print('came here')



            checkpoint = ModelCheckpoint(NAME + '.hdf5',  # model filename

                                         monitor='val_loss', # quantity to monitor

                                         verbose=0, # verbosity - 0 or 1

                                         save_best_only= True, # The latest best model will not be overwritten

                                         mode='auto') # The decision to overwrite model is made 

                                                      # automatically depending on the quantity to monitor



            model.compile(loss='binary_crossentropy',

                            optimizer=opt,

                            metrics=['accuracy'])



            # model_details = model.fit(X, y,

            #                     batch_size = 32,

            #                     epochs = 12, # number of iterations

            #                     validation_split=0.2,

            #                     callbacks=[checkpoint],

            #                     verbose=1)





            # --------------

            earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement

                                      min_delta = 0, #Abs value and is the min change required before we stop

                                      patience = 15, #Number of epochs we wait before stopping 

                                      verbose = 1,

                                      restore_best_weights = True) #keeps the best weigths once stopped



            ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)

            callbacks = [earlystop, checkpoint, ReduceLR]



            history = model.fit_generator(datagen.flow(x_train, y_train, batch_size= 32), epochs = 20, verbose=1,callbacks = callbacks,validation_data=(x_valid,y_validate))







            # model.fit(X, y,

            #             batch_size=32,

            #             epochs=10,

            #             validation_split=0.2)



            model.save(NAME + '.hdf5')
import pickle



pickle_out = open("Trained_cnn_history_my_model.pickle","wb")

pickle.dump(history.history, pickle_out)

pickle_out.close()
pickle_in = open("Trained_cnn_history_my_model.pickle","rb")

saved_history = pickle.load(pickle_in)

print(saved_history)
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()
model = tf.keras.models.load_model(NAME + '.hdf5')

# model.load_weights('128x3-CNN.hdf5')



score = model.evaluate(x_test,y_test,verbose=0)

print('Test Loss :',score[0])

print('Test Accuracy :',score[1])



predicted_classes = model.predict_classes(x_test)

confusion_mtx = confusion_matrix(y_test, predicted_classes) 



plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('confusion_matrix')

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ['R','O'], rotation=90)

plt.yticks(tick_marks, ['R','O'])

#Following is to mention the predicated numbers in the plot and highligh the numbers the most predicted number for particular label

thresh = confusion_mtx.max() / 2.

for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):

    plt.text(j, i, confusion_mtx[i, j],

    horizontalalignment="center",

    color="white" if confusion_mtx[i, j] > thresh else "black")



plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

print(x_train.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf

import pickle

import time

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint

import keras





# pickle_in = open("X.pickle", "rb")

# X = pickle.load(pickle_in)



# pickle_in = open("y.pickle", "rb")

# y = pickle.load(pickle_in)



# X = X/255.0

IMG_SIZE = 100

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)



# y = np.array(y)



opt = tf.keras.optimizers.Adam(learning_rate=0.001)



dense_layers = [2]

layer_sizes = [128]

conv_layers = [3]

for dense_layer in dense_layers:

    for layer_size in layer_sizes:

        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(

                conv_layer, layer_size, dense_layer, int(time.time()))

            print(NAME)



            model = Sequential()



            model.add(Conv2D(layer_size, (3, 3), input_shape=(100,100,3)))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))



            for l in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3, 3)))

                model.add(Activation('relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))



            model.add(Flatten())



            for _ in range(dense_layer):

                model.add(Dense(layer_size))

                model.add(Activation('relu'))

                model.add(Dropout(0.5))



            model.add(Dense(1))

            model.add(Activation('sigmoid'))

#             print('came here')



            checkpoint = ModelCheckpoint(NAME + '.hdf5',  # model filename

                                         monitor='val_loss', # quantity to monitor

                                         verbose=0, # verbosity - 0 or 1

                                         save_best_only= True, # The latest best model will not be overwritten

                                         mode='auto') # The decision to overwrite model is made 

                                                      # automatically depending on the quantity to monitor



            model.compile(loss='binary_crossentropy',

                            optimizer=opt,

                            metrics=['accuracy'])



            # model_details = model.fit(X, y,

            #                     batch_size = 32,

            #                     epochs = 12, # number of iterations

            #                     validation_split=0.2,

            #                     callbacks=[checkpoint],

            #                     verbose=1)





            # --------------

            earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement

                                      min_delta = 0, #Abs value and is the min change required before we stop

                                      patience = 15, #Number of epochs we wait before stopping 

                                      verbose = 1,

                                      restore_best_weights = True) #keeps the best weigths once stopped



            ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)

            callbacks = [earlystop, checkpoint, ReduceLR]



            history = model.fit_generator(datagen.flow(x_train, y_train, batch_size= 32), epochs = 20, verbose=1,callbacks = callbacks,validation_data=(x_valid,y_validate))







            # model.fit(X, y,

            #             batch_size=32,

            #             epochs=10,

            #             validation_split=0.2)



            model.save(NAME + '.hdf5')
import pickle



pickle_out = open("Trained_cnn_history_my_model.pickle","wb")

pickle.dump(history.history, pickle_out)

pickle_out.close()
pickle_in = open("Trained_cnn_history_my_model.pickle","rb")

saved_history = pickle.load(pickle_in)

print(saved_history)
model = tf.keras.models.load_model(NAME + '.hdf5')

# model.load_weights('128x3-CNN.hdf5')



score = model.evaluate(x_test,y_test,verbose=0)

print('Test Loss :',score[0])

print('Test Accuracy :',score[1])



predicted_classes = model.predict_classes(x_test)
import matplotlib.pyplot as plt



# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validate'], loc='upper left')

plt.show()