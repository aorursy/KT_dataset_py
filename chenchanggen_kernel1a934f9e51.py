# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

#from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, TensorBoard



#load the data

train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

dig_test = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

# print(train)

Y_train = train["label"] 

Y_dig = dig_test["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1)  

test = test.drop(labels=['id'], axis=1) 

dig_test = dig_test.drop(labels = ["label"],axis = 1)

del train



print(Y_train.value_counts())



# Normalize the data

X_train = X_train / 255.0

test = test / 255.0

dig_test = dig_test / 255.0



# Reshape the image data (height = 28px, width = 28px , chanal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

dig_test = dig_test.values.reshape(-1,28,28,1)



Y_train = to_categorical(Y_train, num_classes = 10)  

Y_dig = to_categorical(Y_dig, num_classes=10)



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)





# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',

                 activation ='relu'))  #11月21日加

model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))  #11月21日加

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',

                 activation ='relu'))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(32, activation = "relu"))

model.add(Dense(10, activation = "softmax"))





# Compile the model

model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])



# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',

                                            patience=3,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)

epochs = 30 # larger epochs will be needed for higher accuracy

batch_size = 124



TensorBoardcallback = TensorBoard(

    log_dir='./logs',

    histogram_freq=1, batch_size=32,

    write_graph=True, write_grads=False, write_images=True,

    embeddings_freq=0, embeddings_layer_names=None,

    embeddings_metadata=None, embeddings_data=None, update_freq=epochs

)



datagen = ImageDataGenerator(

       

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        ) 



datagen.fit(X_train)



#Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction, TensorBoardcallback],use_multiprocessing=False)



fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

#legend = ax[0].legend(loc='best', shadow=True)

ax[0].legend(loc='best', shadow=False)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

#legend = ax[1].legend(loc='best', shadow=True)

ax[1].legend(loc='best', shadow=True)

plt.show()
# Test on the Dig-MNIST data

dig_result = model.evaluate(dig_test, Y_dig, verbose=0)

print('loss of the model on Dig-MNIST:', dig_result[0])

print('Accuracy of the model on Dig-MNIST:', dig_result[1])

# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



#results = pd.Series(results,name="label")



submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = results

submission.to_csv("submission.csv", index=False)