# Fundamental libraries

import numpy as np

import pandas as pd



# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# General ML libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import operator

import time



# Neural networks libraries

import keras

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")



print("Original data structure:")

display(train.head())
fig = sns.countplot(train['label'], alpha=0.75).set_title("Digit counts")

plt.xlabel("Digits")

plt.ylabel("Counts")

plt.savefig('digit_counts.png')

plt.show()
train.isna().sum().sort_values(ascending=False)
img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw, test):

    y = raw["label"]

    x = raw.drop(labels = ["label"],axis = 1) 

    

    x = x/255.

    x = x.values.reshape(-1, img_rows,img_cols,1)

    

    test = test/255.

    test = test.values.reshape(-1,img_rows,img_cols,1)

    

    return x, y, test



X_train, Y_train, X_test = prep_data(train, test)

Y_train = to_categorical(Y_train, num_classes)



print("Data preparation correctly finished")
batch_size = 64



model_1 = Sequential()

model_1.add(Conv2D(filters=16, kernel_size=(4,4),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_1.add(MaxPool2D())

model_1.add(Flatten())

model_1.add(Dense(256, activation='relu'))

model_1.add(Dense(num_classes, activation='softmax'))



print("CNN ready to compile")
model_1.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



history = model_1.fit(X_train, Y_train,

          batch_size=batch_size,

          epochs=20,

          validation_split = 0.1)



print("Fitting finished")
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model_1 accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('initial_cnn.png')

plt.show()
# predict results

results = model_1.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step3.csv",index=False)
# Model_kernel_1: 3x3

model_kernel_1 = Sequential()

model_kernel_1.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',

                     activation='relu', 

                     input_shape=(img_rows, img_cols, 1)))

model_kernel_1.add(MaxPool2D(padding='same'))

model_kernel_1.add(Flatten())

model_kernel_1.add(Dense(256, activation='relu'))

model_kernel_1.add(Dense(num_classes, activation='softmax'))

    

# Model_kernel_2: 4x4

model_kernel_2 = Sequential()

model_kernel_2.add(Conv2D(filters=16, kernel_size=(4,4), padding='same',

                     activation='relu', 

                     input_shape=(img_rows, img_cols, 1)))

model_kernel_2.add(MaxPool2D(padding='same'))

model_kernel_2.add(Flatten())

model_kernel_2.add(Dense(256, activation='relu'))

model_kernel_2.add(Dense(num_classes, activation='softmax'))

    

# Model_kernel_3: 5x5

model_kernel_3 = Sequential()

model_kernel_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',

                     activation='relu', 

                     input_shape=(img_rows, img_cols, 1)))

model_kernel_3.add(MaxPool2D(padding='same'))

model_kernel_3.add(Flatten())

model_kernel_3.add(Dense(256, activation='relu'))

model_kernel_3.add(Dense(num_classes, activation='softmax'))



# Model_kernel_4: 6x6

model_kernel_4 = Sequential()

model_kernel_4.add(Conv2D(filters=16, kernel_size=(6,6), padding='same',

                     activation='relu', 

                     input_shape=(img_rows, img_cols, 1)))

model_kernel_4.add(MaxPool2D(padding='same'))

model_kernel_4.add(Flatten())

model_kernel_4.add(Dense(256, activation='relu'))

model_kernel_4.add(Dense(num_classes, activation='softmax')) 
ts = time.time()



n_reps = 10

n_epochs = 20



# Keep track of the history evolution for all repetitions of the CNNs

history_kernel_1, history_kernel_val_1 = [0]*n_epochs, [0]*n_epochs

history_kernel_2, history_kernel_val_2 = [0]*n_epochs, [0]*n_epochs

history_kernel_3, history_kernel_val_3 = [0]*n_epochs, [0]*n_epochs

history_kernel_4, history_kernel_val_4 = [0]*n_epochs, [0]*n_epochs





for rep in range(n_reps):



    # Compile model_kernel_1

    model_kernel_1.compile(loss=keras.losses.categorical_crossentropy,

                optimizer='adam',

                metrics=['accuracy'])

    model_kernel_1_history_rep = model_kernel_1.fit(X_train, Y_train,

            batch_size=batch_size,

            epochs=n_epochs,

            validation_split = 0.1, 

            verbose=0)

    history_kernel_1 = tuple(map(operator.add, history_kernel_1, model_kernel_1_history_rep.history['accuracy']))

    history_kernel_val_1 = tuple(map(operator.add, history_kernel_val_1, model_kernel_1_history_rep.history['val_accuracy']))



    # Compile model_kernel_2

    model_kernel_2.compile(loss=keras.losses.categorical_crossentropy,

                optimizer='adam',

                metrics=['accuracy'])

    model_kernel_2_history_rep = model_kernel_2.fit(X_train, Y_train,

            batch_size=batch_size,

            epochs=n_epochs,

            validation_split = 0.1, 

            verbose=0)

    history_kernel_2 = tuple(map(operator.add, history_kernel_2, model_kernel_2_history_rep.history['accuracy']))

    history_kernel_val_2 = tuple(map(operator.add, history_kernel_val_2, model_kernel_2_history_rep.history['val_accuracy']))

    

    # Compile model_kernel_3

    model_kernel_3.compile(loss=keras.losses.categorical_crossentropy,

                optimizer='adam',

                metrics=['accuracy'])

    model_kernel_3_history_rep = model_kernel_3.fit(X_train, Y_train,

            batch_size=batch_size,

            epochs=n_epochs,

            validation_split = 0.1, 

            verbose=0)

    history_kernel_3 = tuple(map(operator.add, history_kernel_3, model_kernel_3_history_rep.history['accuracy']))

    history_kernel_val_3 = tuple(map(operator.add, history_kernel_val_3, model_kernel_3_history_rep.history['val_accuracy']))

    

    # Compile model_kernel_4

    model_kernel_4.compile(loss=keras.losses.categorical_crossentropy,

                optimizer='adam',

                metrics=['accuracy'])

    model_kernel_4_history_rep = model_kernel_4.fit(X_train, Y_train,

            batch_size=batch_size,

            epochs=n_epochs,

            validation_split = 0.1, 

            verbose=0)

    history_kernel_4 = tuple(map(operator.add, history_kernel_4, model_kernel_4_history_rep.history['accuracy']))

    history_kernel_val_4 = tuple(map(operator.add, history_kernel_val_4, model_kernel_4_history_rep.history['val_accuracy']))    

    

# Average historic data for each CNN (train and valuation)

history_kernel_1 = [x/n_reps for x in list(history_kernel_1)] 

history_kernel_2 = [x/n_reps for x in list(history_kernel_2)]

history_kernel_3 = [x/n_reps for x in list(history_kernel_3)]

history_kernel_4 = [x/n_reps for x in list(history_kernel_4)]

history_kernel_val_1 = [x/n_reps for x in list(history_kernel_val_1)]

history_kernel_val_2 = [x/n_reps for x in list(history_kernel_val_2)]

history_kernel_val_3 = [x/n_reps for x in list(history_kernel_val_3)]

history_kernel_val_4 = [x/n_reps for x in list(history_kernel_val_4)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_kernel_val_1)

plt.plot(history_kernel_val_2)

plt.plot(history_kernel_val_3)

plt.plot(history_kernel_val_4)

plt.title('Model accuracy for different convolution kernel sizes')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

#plt.ylim(0.95,1)

plt.xlim(0,n_epochs)

plt.legend(['3x3', '4x4', '5x5', '6x6'], loc='upper left')

plt.savefig('convolution_kernel_size.png')

plt.show()
# Model_layers_1: 1 Conv2d layer, same as our initial model (model_1) 



# Model_layers_2: 2 Conv2D layers

model_layers_2 = Sequential()

model_layers_2.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_layers_2.add(MaxPool2D(padding='same'))

model_layers_2.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

model_layers_2.add(MaxPool2D(padding='same'))

model_layers_2.add(Flatten())

model_layers_2.add(Dense(256, activation='relu'))

model_layers_2.add(Dense(num_classes, activation='softmax'))



# Model_layers_3: 3 Conv2D layers

model_layers_3 = Sequential()

model_layers_3.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_layers_3.add(MaxPool2D())

model_layers_3.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

model_layers_3.add(MaxPool2D(padding='same'))

model_layers_3.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))

model_layers_3.add(MaxPool2D(padding='same'))

model_layers_3.add(Flatten())

model_layers_3.add(Dense(256, activation='relu'))

model_layers_3.add(Dense(num_classes, activation='softmax'))
n_reps = 5

n_epochs = 20



# Keep track of the history evolution for all repetitions of the CNNs

history_layers_1, history_layers_val_1 = [0]*n_epochs, [0]*n_epochs

history_layers_2, history_layers_val_2 = [0]*n_epochs, [0]*n_epochs

history_layers_3, history_layers_val_3 = [0]*n_epochs, [0]*n_epochs



ts = time.time()



for rep in range(n_reps):



    # Compite model_1

    model_1.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



    history_layers_1_rep = model_1.fit(X_train, Y_train,

          batch_size=batch_size,

          epochs=n_epochs,

          validation_split = 0.1, 

          verbose=0)

    

    history_layers_1 = tuple(map(operator.add, history_layers_1, history_layers_1_rep.history['accuracy']))

    history_layers_val_1 = tuple(map(operator.add, history_layers_val_1, history_layers_1_rep.history['val_accuracy']))

    



    # Compile model_2

    model_layers_2.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_layers_2_rep = model_layers_2.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_layers_2 = tuple(map(operator.add, history_layers_2, history_layers_2_rep.history['accuracy']))

    history_layers_val_2 = tuple(map(operator.add, history_layers_val_2, history_layers_2_rep.history['val_accuracy']))



    

    # Compile model_3

    model_layers_3.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_layers_3_rep = model_layers_3.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_layers_3 = tuple(map(operator.add, history_layers_3, history_layers_3_rep.history['accuracy']))

    history_layers_val_3 = tuple(map(operator.add, history_layers_val_3, history_layers_3_rep.history['val_accuracy']))

    

# Average historic data for each CNN (train and valuation)

history_layers_1 = [x/n_reps for x in list(history_layers_1)] 

history_layers_2 = [x/n_reps for x in list(history_layers_2)]

history_layers_3 = [x/n_reps for x in list(history_layers_3)]

history_layers_val_1 = [x/n_reps for x in list(history_layers_val_1)]

history_layers_val_2 = [x/n_reps for x in list(history_layers_val_2)]

history_layers_val_3 = [x/n_reps for x in list(history_layers_val_3)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_layers_val_1)

plt.plot(history_layers_val_2)

plt.plot(history_layers_val_3)

plt.title('Model accuracy for different number of Conv layers')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.98,1)

plt.xlim(0,20)

plt.legend(['1 layer', '2 layers', '3 layers'], loc='upper left')

plt.savefig('number_of_layers.png')

plt.show()
# Model_size_1: 8-16

model_size_1 = Sequential()

model_size_1.add(Conv2D(filters=16, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_size_1.add(MaxPool2D())

model_size_1.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model_size_1.add(MaxPool2D(padding='same'))

model_size_1.add(Flatten())

model_size_1.add(Dense(256, activation='relu'))

model_size_1.add(Dense(num_classes, activation='softmax'))



# Model_size_2: 16-32

model_size_2 = Sequential()

model_size_2.add(Conv2D(filters=16, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_size_2.add(MaxPool2D())

model_size_2.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model_size_2.add(MaxPool2D(padding='same'))

model_size_2.add(Flatten())

model_size_2.add(Dense(256, activation='relu'))

model_size_2.add(Dense(num_classes, activation='softmax'))



# Model_size_3: 32-32

model_size_3 = Sequential()

model_size_3.add(Conv2D(filters=16, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_size_3.add(MaxPool2D())

model_size_3.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model_size_3.add(MaxPool2D(padding='same'))

model_size_3.add(Flatten())

model_size_3.add(Dense(256, activation='relu'))

model_size_3.add(Dense(num_classes, activation='softmax'))



# Model_size_4: 24-48

model_size_4 = Sequential()

model_size_4.add(Conv2D(filters=24, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_size_4.add(MaxPool2D())

model_size_4.add(Conv2D(filters=48, kernel_size=(5,5), activation='relu'))

model_size_4.add(MaxPool2D(padding='same'))

model_size_4.add(Flatten())

model_size_4.add(Dense(256, activation='relu'))

model_size_4.add(Dense(num_classes, activation='softmax'))



# Model_size_5: 32-64

model_size_5 = Sequential()

model_size_5.add(Conv2D(filters=32, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_size_5.add(MaxPool2D())

model_size_5.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))

model_size_5.add(MaxPool2D(padding='same'))

model_size_5.add(Flatten())

model_size_5.add(Dense(256, activation='relu'))

model_size_5.add(Dense(num_classes, activation='softmax'))



# Model_size_6: 48-96

model_size_6 = Sequential()

model_size_6.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_size_6.add(MaxPool2D())

model_size_6.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_size_6.add(MaxPool2D(padding='same'))

model_size_6.add(Flatten())

model_size_6.add(Dense(256, activation='relu'))

model_size_6.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 3

n_epochs = 20



# Keep track of the history evolution for all repetitions of the CNNs

history_size_1, history_size_val_1 = [0]*n_epochs, [0]*n_epochs

history_size_2, history_size_val_2 = [0]*n_epochs, [0]*n_epochs

history_size_3, history_size_val_3 = [0]*n_epochs, [0]*n_epochs

history_size_4, history_size_val_4 = [0]*n_epochs, [0]*n_epochs

history_size_5, history_size_val_5 = [0]*n_epochs, [0]*n_epochs

history_size_6, history_size_val_6 = [0]*n_epochs, [0]*n_epochs





for rep in range(n_reps):



    # Compite model_1

    model_size_1.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



    history_size_1_rep = model_size_1.fit(X_train, Y_train,

          batch_size=batch_size,

          epochs=n_epochs,

          validation_split = 0.1, 

          verbose=0)

    

    history_size_1 = tuple(map(operator.add, history_size_1, history_size_1_rep.history['accuracy']))

    history_size_val_1 = tuple(map(operator.add, history_size_val_1, history_size_1_rep.history['val_accuracy']))

    



    # Compile model_2

    model_size_2.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_size_2_rep = model_size_2.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_size_2 = tuple(map(operator.add, history_size_2, history_size_2_rep.history['accuracy']))

    history_size_val_2 = tuple(map(operator.add, history_size_val_2, history_size_2_rep.history['val_accuracy']))



    

    # Compile model_3

    model_size_3.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_size_3_rep = model_size_3.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_size_3 = tuple(map(operator.add, history_size_3, history_size_3_rep.history['accuracy']))

    history_size_val_3 = tuple(map(operator.add, history_size_val_3, history_size_3_rep.history['val_accuracy']))

    

    # Compile model_4

    model_size_4.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_size_4_rep = model_size_4.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_size_4 = tuple(map(operator.add, history_size_4, history_size_4_rep.history['accuracy']))

    history_size_val_4 = tuple(map(operator.add, history_size_val_4, history_size_4_rep.history['val_accuracy']))

    

    # Compile model_5

    model_size_5.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_size_5_rep = model_size_5.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_size_5 = tuple(map(operator.add, history_size_5, history_size_5_rep.history['accuracy']))

    history_size_val_5 = tuple(map(operator.add, history_size_val_5, history_size_5_rep.history['val_accuracy']))

    

    # Compile model_6

    model_size_6.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_size_6_rep = model_size_6.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_size_6 = tuple(map(operator.add, history_size_6, history_size_6_rep.history['accuracy']))

    history_size_val_6 = tuple(map(operator.add, history_size_val_6, history_size_6_rep.history['val_accuracy']))

    

    

# Average historic data for each CNN (train and valuation)

history_size_1 = [x/n_reps for x in list(history_size_1)] 

history_size_2 = [x/n_reps for x in list(history_size_2)]

history_size_3 = [x/n_reps for x in list(history_size_3)]

history_size_4 = [x/n_reps for x in list(history_size_4)] 

history_size_5 = [x/n_reps for x in list(history_size_5)]

history_size_6 = [x/n_reps for x in list(history_size_6)]

history_size_val_1 = [x/n_reps for x in list(history_size_val_1)]

history_size_val_2 = [x/n_reps for x in list(history_size_val_2)]

history_size_val_3 = [x/n_reps for x in list(history_size_val_3)]

history_size_val_4 = [x/n_reps for x in list(history_size_val_4)]

history_size_val_5 = [x/n_reps for x in list(history_size_val_5)]

history_size_val_6 = [x/n_reps for x in list(history_size_val_6)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_size_val_1)

plt.plot(history_size_val_2)

plt.plot(history_size_val_3)

plt.plot(history_size_val_4)

plt.plot(history_size_val_5)

plt.plot(history_size_val_6)

plt.title('Model accuracy for different Conv sizes')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.98,1)

plt.xlim(0,n_epochs)

plt.legend(['8-16', '16-32', '32-32', '24-48', '32-64', '48-96', '64,128'], loc='upper left')

plt.savefig('convolution_size.png')

plt.show()
# predict results

results = model_size_6.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step6.csv",index=False)
# Model_dropout_1: No dropout, same as model_size_6



# Model_dropout_2: 20% dropout

model_dropout_2 = Sequential()

model_dropout_2.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dropout_2.add(MaxPool2D())

model_dropout_2.add(Dropout(0.2))

model_dropout_2.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dropout_2.add(MaxPool2D(padding='same'))

model_dropout_2.add(Dropout(0.2))

model_dropout_2.add(Flatten())

model_dropout_2.add(Dense(256, activation='relu'))

model_dropout_2.add(Dense(num_classes, activation='softmax'))



# Model_dropout_3: 40% dropout

model_dropout_3 = Sequential()

model_dropout_3.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dropout_3.add(MaxPool2D())

model_dropout_3.add(Dropout(0.4))

model_dropout_3.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dropout_3.add(MaxPool2D(padding='same'))

model_dropout_3.add(Dropout(0.4))

model_dropout_3.add(Flatten())

model_dropout_3.add(Dense(256, activation='relu'))

model_dropout_3.add(Dense(num_classes, activation='softmax'))



# Model_dropout_4: 60% dropout

model_dropout_4 = Sequential()

model_dropout_4.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dropout_4.add(MaxPool2D())

model_dropout_4.add(Dropout(0.6))

model_dropout_4.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dropout_4.add(MaxPool2D(padding='same'))

model_dropout_4.add(Dropout(0.6))

model_dropout_4.add(Flatten())

model_dropout_4.add(Dense(256, activation='relu'))

model_dropout_4.add(Dense(num_classes, activation='softmax'))



# Model_dropout_5: 80% dropout

model_dropout_5 = Sequential()

model_dropout_5.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dropout_5.add(MaxPool2D())

model_dropout_5.add(Dropout(0.8))

model_dropout_5.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dropout_5.add(MaxPool2D(padding='same'))

model_dropout_5.add(Dropout(0.8))

model_dropout_5.add(Flatten())

model_dropout_5.add(Dense(256, activation='relu'))

model_dropout_5.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 3

n_epochs = 20



# Keep track of the history evolution for all repetitions of the CNNs

history_dropout_1, history_dropout_val_1 = [0]*n_epochs, [0]*n_epochs

history_dropout_2, history_dropout_val_2 = [0]*n_epochs, [0]*n_epochs

history_dropout_3, history_dropout_val_3 = [0]*n_epochs, [0]*n_epochs

history_dropout_4, history_dropout_val_4 = [0]*n_epochs, [0]*n_epochs

history_dropout_5, history_dropout_val_5 = [0]*n_epochs, [0]*n_epochs





for rep in range(n_reps):



    # Model_1 was previously computed in Step 6



    # Compile model_2

    model_dropout_2.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dropout_2_rep = model_dropout_2.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dropout_2 = tuple(map(operator.add, history_dropout_2, history_dropout_2_rep.history['accuracy']))

    history_dropout_val_2 = tuple(map(operator.add, history_dropout_val_2, history_dropout_2_rep.history['val_accuracy']))



    

    # Compile model_3

    model_dropout_3.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dropout_3_rep = model_dropout_3.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dropout_3 = tuple(map(operator.add, history_dropout_3, history_dropout_3_rep.history['accuracy']))

    history_dropout_val_3 = tuple(map(operator.add, history_dropout_val_3, history_dropout_3_rep.history['val_accuracy']))

    

    # Compile model_4

    model_dropout_4.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dropout_4_rep = model_dropout_4.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dropout_4 = tuple(map(operator.add, history_dropout_4, history_dropout_4_rep.history['accuracy']))

    history_dropout_val_4 = tuple(map(operator.add, history_dropout_val_4, history_dropout_4_rep.history['val_accuracy']))

    

    # Compile model_5

    model_dropout_5.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dropout_5_rep = model_dropout_5.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dropout_5 = tuple(map(operator.add, history_dropout_5, history_dropout_5_rep.history['accuracy']))

    history_dropout_val_5 = tuple(map(operator.add, history_dropout_val_5, history_dropout_5_rep.history['val_accuracy']))

       

    

# Average historic data for each CNN (train and valuation)

history_dropout_2 = [x/n_reps for x in list(history_dropout_2)]

history_dropout_3 = [x/n_reps for x in list(history_dropout_3)]

history_dropout_4 = [x/n_reps for x in list(history_dropout_4)] 

history_dropout_5 = [x/n_reps for x in list(history_dropout_5)]

history_dropout_val_2 = [x/n_reps for x in list(history_dropout_val_2)]

history_dropout_val_3 = [x/n_reps for x in list(history_dropout_val_3)]

history_dropout_val_4 = [x/n_reps for x in list(history_dropout_val_4)]

history_dropout_val_5 = [x/n_reps for x in list(history_dropout_val_5)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_size_val_1)

plt.plot(history_dropout_val_2)

plt.plot(history_dropout_val_3)

plt.plot(history_dropout_val_4)

plt.plot(history_dropout_val_5)

plt.title('Model accuracy for different dropouts')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.98,1)

plt.xlim(0,n_epochs)

plt.legend(['0% dropout', '20% dropout', '40% dropout', '60% dropout', '80% dropout'], loc='upper left')

plt.savefig('dropout.png')

plt.show()
# predict results

results = model_dropout_3.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step7.csv",index=False)
# Model_dense_64: 20% dropout

model_dense_1 = Sequential()

model_dense_1.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dense_1.add(MaxPool2D())

model_dense_1.add(Dropout(0.4))

model_dense_1.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dense_1.add(MaxPool2D(padding='same'))

model_dense_1.add(Dropout(0.4))

model_dense_1.add(Flatten())

model_dense_1.add(Dense(64, activation='relu'))

model_dense_1.add(Dense(num_classes, activation='softmax'))



# Model_dense_128: 128 nodes dense layer

model_dense_2 = Sequential()

model_dense_2.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dense_2.add(MaxPool2D())

model_dense_2.add(Dropout(0.4))

model_dense_2.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dense_2.add(MaxPool2D(padding='same'))

model_dense_2.add(Dropout(0.4))

model_dense_2.add(Flatten())

model_dense_2.add(Dense(128, activation='relu'))

model_dense_2.add(Dense(num_classes, activation='softmax'))



# Model_dense_3: 256 nodes dense layer. Same as model from Step 7.



# Model_dense_4: 512 nodes dense layer

model_dense_4 = Sequential()

model_dense_4.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dense_4.add(MaxPool2D())

model_dense_4.add(Dropout(0.4))

model_dense_4.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dense_4.add(MaxPool2D(padding='same'))

model_dense_4.add(Dropout(0.4))

model_dense_4.add(Flatten())

model_dense_4.add(Dense(512, activation='relu'))

model_dense_4.add(Dense(num_classes, activation='softmax'))



# Model_dense_5: 1024 nodes dense layer

model_dense_5 = Sequential()

model_dense_5.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_dense_5.add(MaxPool2D())

model_dense_5.add(Dropout(0.4))

model_dense_5.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_dense_5.add(MaxPool2D(padding='same'))

model_dense_5.add(Dropout(0.4))

model_dense_5.add(Flatten())

model_dense_5.add(Dense(1024, activation='relu'))

model_dense_5.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 3

n_epochs = 20



# Keep track of the history evolution for all repetitions of the CNNs

history_dense_1, history_dense_val_1 = [0]*n_epochs, [0]*n_epochs

history_dense_2, history_dense_val_2 = [0]*n_epochs, [0]*n_epochs

history_dense_4, history_dense_val_4 = [0]*n_epochs, [0]*n_epochs

history_dense_5, history_dense_val_5 = [0]*n_epochs, [0]*n_epochs





for rep in range(n_reps):



    # Compile model_dense_1

    model_dense_1.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dense_1_rep = model_dense_1.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dense_1 = tuple(map(operator.add, history_dense_1, history_dense_1_rep.history['accuracy']))

    history_dense_val_1 = tuple(map(operator.add, history_dense_val_1, history_dense_1_rep.history['val_accuracy']))



    # Compile model_dense_2

    model_dense_2.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dense_2_rep = model_dense_2.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dense_2 = tuple(map(operator.add, history_dense_2, history_dense_2_rep.history['accuracy']))

    history_dense_val_2 = tuple(map(operator.add, history_dense_val_2, history_dense_2_rep.history['val_accuracy']))

    

    # Model with 256 dense nodes was compiled in Step 7.

    

    # Compile model_dense_4

    model_dense_4.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dense_4_rep = model_dense_4.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dense_4 = tuple(map(operator.add, history_dense_4, history_dense_4_rep.history['accuracy']))

    history_dense_val_4 = tuple(map(operator.add, history_dense_val_4, history_dense_4_rep.history['val_accuracy']))

    

    # Compile model_dense_5

    model_dense_5.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])



    history_dense_5_rep = model_dense_5.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=n_epochs,

              validation_split = 0.1, 

              verbose=0)

    

    history_dense_5 = tuple(map(operator.add, history_dense_5, history_dense_5_rep.history['accuracy']))

    history_dense_val_5 = tuple(map(operator.add, history_dense_val_5, history_dense_5_rep.history['val_accuracy']))

       

    

# Average historic data for each CNN (train and valuation)

history_dense_1 = [x/n_reps for x in list(history_dense_2)]

history_dense_2 = [x/n_reps for x in list(history_dense_2)]

history_dense_4 = [x/n_reps for x in list(history_dense_4)] 

history_dense_5 = [x/n_reps for x in list(history_dense_5)]

history_dense_val_1 = [x/n_reps for x in list(history_dense_val_2)]

history_dense_val_2 = [x/n_reps for x in list(history_dense_val_2)]

history_dense_val_4 = [x/n_reps for x in list(history_dense_val_4)]

history_dense_val_5 = [x/n_reps for x in list(history_dense_val_5)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_dense_val_1)

plt.plot(history_dense_val_2)

plt.plot(history_dropout_val_3)

plt.plot(history_dense_val_4)

plt.plot(history_dense_val_5)

plt.title('Model accuracy for different number of dense nodes')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.99,1)

plt.xlim(0,n_epochs)

plt.legend(['64 dense nodes', '128 dense nodes', '256 dense nodes', '512 dense nodes', '1024 dense nodes'], loc='upper left')

plt.savefig('dense_nodes.png')

plt.show()
# predict results

results = model_dropout_3.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step8.csv",index=False)
X_train_validation, X_val_validation, Y_train_validation, Y_val_validation = train_test_split(X_train, Y_train, test_size = 0.2)



# Generate augmented additional data

data_generator_with_aug = ImageDataGenerator(width_shift_range = 0.1,

                                   height_shift_range = 0.1,

                                   rotation_range = 10,

                                   zoom_range = 0.1)

data_generator_no_aug = ImageDataGenerator()



train_generator = data_generator_with_aug.flow(X_train_validation, Y_train_validation, batch_size=64)

validation_generator = data_generator_no_aug.flow(X_train_validation, Y_train_validation, batch_size=64)
# Model for augmented data (same as dropout_3)

model_augmentation = Sequential()

model_augmentation.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_augmentation.add(MaxPool2D())

model_augmentation.add(Dropout(0.4))

model_augmentation.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_augmentation.add(MaxPool2D(padding='same'))

model_augmentation.add(Dropout(0.4))

model_augmentation.add(Flatten())

model_augmentation.add(Dense(256, activation='relu'))

model_augmentation.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 10

n_epochs = 20



# Use the model with better score and include augmented data. Repeat n_reps times for averaging

history_augmentation, history_augmentation_val = [0]*n_epochs, [0]*n_epochs



for rep in range(n_reps):

    # Compile the model

    model_augmentation.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    # Fit the model

    history_augmentation_rep = model_augmentation.fit_generator(train_generator,

                                                         epochs = n_epochs, 

                                                         steps_per_epoch = X_train_validation.shape[0]//64,

                                                         validation_data = validation_generator,  

                                                         verbose=0)

    history_augmentation = tuple(map(operator.add, history_augmentation, history_augmentation_rep.history['accuracy']))

    history_augmentation_val = tuple(map(operator.add, history_augmentation_val, history_augmentation_rep.history['val_accuracy']))



history_augmentation = [x/n_reps for x in list(history_augmentation)]

history_augmentation_val = [x/n_reps for x in list(history_augmentation_val)]  

    

print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_augmentation_val)

plt.plot(history_dropout_val_3)

plt.title('Model accuracy for data augmentation')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.99,1)

plt.xlim(0,n_epochs)

plt.legend(['with augmentation', 'without augmentation'], loc='upper left')

plt.savefig('augmentation.png')

plt.show()
# predict results

results = model_dropout_3.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step9.csv",index=False)
# Model_batch_norm: Add a batch normalization procedure after each convolution and dense layer

model_batch_norm = Sequential()

model_batch_norm.add(Conv2D(filters=48, kernel_size=(5,5),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_batch_norm.add(BatchNormalization())

model_batch_norm.add(MaxPool2D())

model_batch_norm.add(Dropout(0.4))

model_batch_norm.add(Conv2D(filters=96, kernel_size=(5,5), activation='relu'))

model_batch_norm.add(BatchNormalization())

model_batch_norm.add(MaxPool2D(padding='same'))

model_batch_norm.add(Dropout(0.4))

model_batch_norm.add(Flatten())

model_batch_norm.add(Dense(256, activation='relu'))

model_batch_norm.add(BatchNormalization())

model_batch_norm.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 10

n_epochs = 20



# Use the model with better score and include augmented data. Repeat n_reps times for averaging

history_batch_norm, history_batch_norm_val = [0]*n_epochs, [0]*n_epochs



for rep in range(n_reps):

    # Compile the model

    model_batch_norm.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    # Fit the model

    history_batch_norm_rep = model_batch_norm.fit_generator(train_generator,

                                                         epochs = n_epochs, 

                                                         steps_per_epoch = X_train_validation.shape[0]//64,

                                                         validation_data = validation_generator,  

                                                         verbose=0)

    history_batch_norm = tuple(map(operator.add, history_batch_norm, history_batch_norm_rep.history['accuracy']))

    history_batch_norm_val = tuple(map(operator.add, history_batch_norm_val, history_batch_norm_rep.history['val_accuracy']))

    

history_batch_norm = [x/n_reps for x in list(history_batch_norm)]

history_batch_norm_val = [x/n_reps for x in list(history_batch_norm_val)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_batch_norm_val)

plt.plot(history_augmentation_val)

plt.title('Model accuracy for batch normalization')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.99,1)

plt.xlim(0,n_epochs)

plt.legend(['with batch normalization', 'without batch normalization'], loc='upper left')

plt.savefig('batch_normalization.png')

plt.show()
# predict results

results = model_batch_norm.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step10.csv",index=False)
# Model_batch_norm: Add a batch normalization procedure after each convolution and dense layer

model_smaller_kernels = Sequential()

model_smaller_kernels.add(Conv2D(filters=48, kernel_size=(3,3),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_smaller_kernels.add(BatchNormalization())

model_smaller_kernels.add(Conv2D(filters=46, kernel_size=(3,3), activation='relu'))

model_smaller_kernels.add(BatchNormalization())

model_smaller_kernels.add(MaxPool2D())

model_smaller_kernels.add(Dropout(0.4))

model_smaller_kernels.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))

model_smaller_kernels.add(BatchNormalization())

model_smaller_kernels.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))

model_smaller_kernels.add(BatchNormalization())

model_smaller_kernels.add(MaxPool2D(padding='same'))

model_smaller_kernels.add(Dropout(0.4))

model_smaller_kernels.add(Flatten())

model_smaller_kernels.add(Dense(256, activation='relu'))

model_smaller_kernels.add(BatchNormalization())

model_smaller_kernels.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 10

n_epochs = 20



# Use the model with better score and include augmented data. Repeat n_reps times for averaging

history_smaller_kernels, history_smaller_kernels_val = [0]*n_epochs, [0]*n_epochs



for rep in range(n_reps):

    # Compile the model

    model_smaller_kernels.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    # Fit the model

    history_smaller_kernels_rep = model_smaller_kernels.fit_generator(train_generator,

                                                         epochs = n_epochs, 

                                                         steps_per_epoch = X_train_validation.shape[0]//64,

                                                         validation_data = validation_generator,  

                                                         verbose=0)

    history_smaller_kernels = tuple(map(operator.add, history_smaller_kernels, history_smaller_kernels_rep.history['accuracy']))

    history_smaller_kernels_val = tuple(map(operator.add, history_smaller_kernels_val, history_smaller_kernels_rep.history['val_accuracy']))

    

history_smaller_kernels = [x/n_reps for x in list(history_smaller_kernels)]

history_smaller_kernels_val = [x/n_reps for x in list(history_smaller_kernels_val)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_smaller_kernels_val)

plt.plot(history_batch_norm_val)

plt.title('Model accuracy replacing Conv2D(5x5) by Conv2D(3x3)+Conv2D(3x3)')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.99,1)

plt.xlim(0,n_epochs)

plt.legend(['3x3+3x3', '5x5'], loc='upper left')

plt.savefig('replace_big_convs.png')

plt.show()
# predict results

results = model_smaller_kernels.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step11.csv",index=False)
# Model_batch_norm: Add a batch normalization procedure after each convolution and dense layer

model_pool_conv = Sequential()

model_pool_conv.add(Conv2D(filters=48, kernel_size=(3,3),

                 activation='relu', 

                 input_shape=(img_rows, img_cols, 1)))

model_pool_conv.add(BatchNormalization())

model_pool_conv.add(Conv2D(filters=46, kernel_size=(3,3), activation='relu'))

model_pool_conv.add(BatchNormalization())

model_pool_conv.add(Conv2D(filters=46, kernel_size=(5,5), activation='relu', strides=2, padding='same'))

model_pool_conv.add(Dropout(0.4))

model_pool_conv.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))

model_pool_conv.add(BatchNormalization())

model_pool_conv.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))

model_pool_conv.add(BatchNormalization())

model_pool_conv.add(Conv2D(filters=46, kernel_size=(5,5), activation='relu', strides=2, padding='same'))

model_pool_conv.add(Dropout(0.4))

model_pool_conv.add(Flatten())

model_pool_conv.add(Dense(256, activation='relu'))

model_pool_conv.add(BatchNormalization())

model_pool_conv.add(Dense(num_classes, activation='softmax'))
ts = time.time()



n_reps = 10

n_epochs = 20



# Use the model with better score and include augmented data. Repeat n_reps times for averaging

history_pool_conv, history_pool_conv_val = [0]*n_epochs, [0]*n_epochs



for rep in range(n_reps):

    # Compile the model

    model_pool_conv.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    # Fit the model

    history_pool_conv_rep = model_pool_conv.fit_generator(train_generator,

                                                         epochs = n_epochs, 

                                                         steps_per_epoch = X_train_validation.shape[0]//64,

                                                         validation_data = validation_generator,  

                                                         verbose=0)

    history_pool_conv = tuple(map(operator.add, history_pool_conv, history_pool_conv_rep.history['accuracy']))

    history_pool_conv_val = tuple(map(operator.add, history_pool_conv_val, history_pool_conv_rep.history['val_accuracy']))

    

history_pool_conv = [x/n_reps for x in list(history_pool_conv)]

history_pool_conv_val = [x/n_reps for x in list(history_pool_conv_val)]



print ("Time spent, " + str(time.time() - ts) + " s")
# Plot the results

plt.plot(history_pool_conv_val)

plt.plot(history_smaller_kernels_val)

plt.title('Model accuracy replacing MaxPool by Conv2D with strides')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.ylim(0.99,1)

plt.xlim(0,n_epochs)

plt.legend(['Conv2D with strides', 'MaxPool'], loc='upper left')

plt.savefig('replace_maxpool_by_conv.png')

plt.show()
# predict results

results = model_pool_conv.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step12.csv",index=False)
ts = time.time()



n_reps = 1

n_epochs = 40



# Use the model with better score and include augmented data. Repeat n_reps times for averaging

history_definitive, history_definitive_val = [0]*n_epochs, [0]*n_epochs



# Callback function (early stopping)

callback_fcn = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+n_epochs))



for rep in range(n_reps):

    # Compile the model

    model_pool_conv.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer='adam',

                  metrics=['accuracy'])

    

    # Fit the model

    history_definitive_rep = model_pool_conv.fit_generator(train_generator,

                                                         epochs = n_epochs, 

                                                         steps_per_epoch = X_train_validation.shape[0]//64,

                                                         validation_data = validation_generator,  

                                                         callbacks=[callback_fcn],

                                                         verbose=0)

    history_definitive = tuple(map(operator.add, history_definitive, history_definitive_rep.history['accuracy']))

    history_definitive_val = tuple(map(operator.add, history_definitive_val, history_definitive_rep.history['val_accuracy']))

    

history_definitive = [x/n_reps for x in list(history_definitive)]

history_definitive_val = [x/n_reps for x in list(history_definitive_val)]



print ("Time spent, " + str(time.time() - ts) + " s")
# predict results

results = model_pool_conv.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit_step13.csv",index=False)