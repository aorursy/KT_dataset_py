# import necessary libraries



import numpy as np

import keras



# MNIST dataset

from keras.datasets import mnist



# to build the model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



# 

from matplotlib import pyplot as plt



# (train,test) data split

from sklearn.model_selection import train_test_split



# to calculate accuracy

import sklearn.metrics as metrics



# learning rate

from keras.callbacks import LearningRateScheduler



# to create artifical train images

from keras.preprocessing.image import ImageDataGenerator



# global variables

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

# Load the data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# shape of train, test data

print({'train data': X_train.shape, 'test data': X_test.shape})
# Images from the train data set

plt.figure(figsize=(15,10))

x, y = 10, 5

for i in range(50):  

    plt.subplot(y, x, i+1)

    plt.imshow(X_train[i].reshape((28,28)),interpolation='nearest')

plt.show()
# reshape the input image using 'image_data_format()'



print("Image data format:", K.image_data_format())



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)



print("train:", X_train.shape)

print("test:",  X_test.shape)
# check the output of output variable 'Y'

print('train_output:', y_train)

print('test_output:', y_test)
# Change the format of output 'Y' using 'keras.utils.to_categorical'



y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)



# print('train_output:', y_train)

# print('test_output:', y_test)
model = [0,0,0]



for i in range(3):

    

    model[i] = Sequential()    

    model[i].add(Conv2D(32, kernel_size = 5, padding='same', activation = 'relu', input_shape = (28,28,1)))

    model[i].add(MaxPooling2D())



    if i>0:

        model[i].add(Conv2D(48, kernel_size = 5, padding='same', activation = 'relu', input_shape = (28,28,1)))

        model[i].add(MaxPooling2D())    



    if i>1:

        model[i].add(Conv2D(64, kernel_size = 5, padding='same', activation = 'relu', input_shape = (28,28,1)))

        model[i].add(MaxPooling2D(padding='same'))

        

# Now flatten the model and compile 



    model[i].add(Flatten())

    model[i].add(Dense(256, activation='relu'))

    model[i].add(Dense(10, activation='softmax'))

    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# to display the summary of models use "model[i].summary()""



model[2].summary()
# Split the data

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size = 0.3)



# fit the model

history = [0] * 3

names = ["1layer","2layers","3layers"]

epochs = 20

for j in range(3):

    history[j] = model[j].fit(X_train1,y_train1, callbacks=[annealer], batch_size=100, epochs = epochs, validation_data = (X_test1,y_test1), verbose=0)        

    print("Conv2D {0}: Epochs={1:d}, train accuracy={2:.5f}, test accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
# Display the test accuracy against epoch 

plt.figure(figsize=(15,5))

for i in range(3):

    plt.plot(history[i].history['val_accuracy'])  

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(names, loc='upper left')
model = [0,0,0,0]



for i in range(4):

    

    model[i] = Sequential()

    model[i].add(Conv2D((8*2**i), kernel_size = 5, activation = 'relu', input_shape = (28,28,1)))

    model[i].add(MaxPooling2D())



    model[i].add(Conv2D((16*2**i), kernel_size = 5, activation = 'relu', input_shape = (28,28,1)))

    model[i].add(MaxPooling2D())



# Now flatten the model and compile 



    model[i].add(Flatten())

    model[i].add(Dense(256, activation='relu'))

    model[i].add(Dense(10, activation='softmax'))

    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# to display the summary of models use "model[i].summary()""



model[2].summary()
# Split the data

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size = 0.3)



# fit the model

history = [0] * 4

names = ["(8,16) maps","(16,32) maps","(32,64) maps", "(64,128) maps"]

epochs = 20

for j in range(4):

    history[j] = model[j].fit(X_train1,y_train1, callbacks=[annealer], batch_size=100, epochs = epochs, validation_data = (X_test1,y_test1), verbose=0)        

    print("Conv2D {0}: Epochs={1:d}, train accuracy={2:.5f}, test accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
# Display the test accuracy against epoch 

plt.figure(figsize=(15,5))

for i in range(4):

    plt.plot(history[i].history['val_accuracy'])  

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(names, loc='upper left')
model = [0]*5



for i in range(5):



    model[i] = Sequential()

    model[i].add(Conv2D(32, kernel_size = 5, activation = 'relu', input_shape = (28,28,1)))

    model[i].add(MaxPooling2D())



    model[i].add(Conv2D(64, kernel_size = 5, activation = 'relu', input_shape = (28,28,1)))

    model[i].add(MaxPooling2D())

    model[i].add(Flatten())



  # Now iterate over Dense layer

    if i >= 0:

        model[i].add(Dense(32*2**i, activation='relu'))



    model[i].add(Dense(10, activation='softmax'))

    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# to display the summary of models use "model[i].summary()""



model[4].summary()
# Split the data

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size = 0.3)



# fit the model

history = [0] * 5

names = ["32 inputs","64 inputs","128 inputs", "256 inputs", "512 inputs"]

epochs = 20

for j in range(5):

    history[j] = model[j].fit(X_train1,y_train1, callbacks=[annealer], batch_size=100, epochs = epochs, validation_data = (X_test1,y_test1), verbose=0)        

    print("Conv2D {0}: Epochs={1:d}, train accuracy={2:.5f}, test accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
model = [0]*5



for i in range(5):

    

    model[i] = Sequential()

    model[i].add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

    model[i].add(MaxPooling2D())

    model[i].add(Dropout(i*0.1))

    model[i].add(Conv2D(64,kernel_size=5,activation='relu'))

    model[i].add(MaxPooling2D())

    model[i].add(Dropout(i*0.1))

    model[i].add(Flatten())

    model[i].add(Dense(128, activation='relu'))

    model[i].add(Dropout(i*0.1))

    model[i].add(Dense(10, activation='softmax'))

    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Split the data

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size = 0.3)



# fit the model

history = [0] * 5

names = ["0% drop","10% drop","20% drop", "30% drop", "40% drop"]

epochs = 20

for j in range(5):

    history[j] = model[j].fit(X_train1,y_train1, callbacks=[annealer], batch_size=100, epochs = epochs, validation_data = (X_test1,y_test1), verbose=0)        

    print("Conv2D {0}: Epochs={1:d}, train accuracy={2:.5f}, test accuracy={3:.5f}".format(

        names[j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
# Display the test accuracy against epoch 

plt.figure(figsize=(15,5))

for i in range(5):

    plt.plot(history[i].history['val_accuracy'])  

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(names, loc='upper left')
# Build the final model



model = Sequential()

model.add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D())

model.add(Dropout(0.3))

model.add(Conv2D(64,kernel_size=5,activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, batch_size = 100,

          epochs = 20, verbose = 1, validation_data = (X_test, y_test))

# Predict 'X_test'

y_test_pred = np.argmax(model.predict(X_test), axis=-1)



# Convert categorical values of y_test

y_actual = np.argmax(y_test, axis=1)



# Accuracy of test data

print('Accuracy:', metrics.accuracy_score(y_actual,y_test_pred))
# Lets add 3*3 kernel_size instead of 5*5

model = Sequential()



# Start - Add two 3*3 kernel_size

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32,kernel_size=3,activation='relu'))

# End - Add two 3*3 kernel_size



model.add(MaxPooling2D())

model.add(Dropout(0.3))



# Start - Add two 3*3 kernel_size

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(Conv2D(64,kernel_size=3,activation='relu'))

# End - Add two 3*3 kernel_size



model.add(MaxPooling2D())

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, batch_size = 100,

          epochs = 20, verbose = 1, validation_data = (X_test, y_test))
# Predict 'X_test'

y_test_pred = np.argmax(model.predict(X_test), axis=-1)



# Convert categorical values of y_test

y_actual = np.argmax(y_test, axis=1)



# Accuracy of test data

print('Accuracy:', metrics.accuracy_score(y_actual,y_test_pred))
model = Sequential()



model.add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32,kernel_size=5, strides = 2, padding = 'same', activation='relu'))



#model.add(MaxPooling2D())

model.add(Dropout(0.3))



 

model.add(Conv2D(64,kernel_size=5,activation='relu'))

model.add(Conv2D(64,kernel_size=5, strides = 2, padding = 'same', activation='relu'))

 

#model.add(MaxPooling2D())

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, batch_size = 100,

          epochs = 20, verbose = 1, validation_data = (X_test, y_test))
# Predict 'X_test'

y_test_pred = np.argmax(model.predict(X_test), axis=-1)



# Convert categorical values of y_test

y_actual = np.argmax(y_test, axis=1)



# Accuracy of test data

print('Accuracy:', metrics.accuracy_score(y_actual,y_test_pred))
# Image data augmentation is used to expand the training dataset in order to improve the performance and ability of the model to generalize. 



image_data_gen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

# Basic Model



model = Sequential()

model.add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D())

model.add(Dropout(0.3))

model.add(Conv2D(64,kernel_size=5,activation='relu'))

model.add(MaxPooling2D())

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Currently i'm using 'above model' for 'Data Augmentation'



model.fit_generator(image_data_gen.flow(X_train,y_train, batch_size = 100), epochs = 50, 

    steps_per_epoch = X_train.shape[0]//100, callbacks=[annealer], verbose=0)

X_train.shape
# Predict 'X_test'

y_test_pred = np.argmax(model.predict(X_test), axis=-1)



# Convert categorical values of y_test

y_actual = np.argmax(y_test, axis=1)



# Accuracy of test data

print('Accuracy:', metrics.accuracy_score(y_actual,y_test_pred))