import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow import keras

from keras import regularizers

from tensorflow.keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
print("train shape is: " + str(train.shape))

print("test shape is: " + str(test.shape))
test.head()
X = train.drop(['label'], axis = 1)

X_valid = test.drop(['id'], axis = 1)
print("original TRAIN shape: " + str(X.shape))

print("original TEST shape: " + str(X_valid.shape))
# reshape data before input to model

X = X.values/255

X_valid = X_valid.values/255

Y = train['label'].values



# reshape

X = X.reshape(X.shape[0],28,28,1)

X_valid = X_valid.reshape(X_valid.shape[0],28,28,1)

#Y = tf.keras.utils.to_categorical(Y)

                                 



print("X data shape: "+str(X.shape))

print("X_valid data shape: "+str(X_valid.shape))

print("Y data shape: "+str(Y.shape))
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size = 0.2)
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X[i][:,:,0], cmap=plt.cm.binary)

    plt.xlabel(train.label[i])

plt.show()
# CNN architechture

f = 2**2



model = tf.keras.Sequential([

    # layer 1

    tf.keras.layers.Conv2D(f*16,kernel_size=(3,3),padding="same",activation='relu',

                           kernel_initializer='he_uniform', 

                           input_shape=(28,28,1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*16, (3,3), padding='same', 

                           activation ='relu',

                           kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*16, (5,5), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    #tf.keras.layers.Dropout(0.15),

    

    tf.keras.layers.Conv2D(f*32, (3,3), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*32, (3,3), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*32, (5,5), padding='same', activation ='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.15),

    

    # layer 3

    tf.keras.layers.Conv2D(f*64,kernel_size=(3,3),padding="same",activation='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(f*64,kernel_size=(3,3),padding="same",activation='relu',

                          kernel_regularizer=regularizers.l2(0.01)),

    #tf.keras.layers.Conv2D(f*64,kernel_size=(5,5),padding="same",activation='relu'),

    #tf.keras.layers.Conv2D(f*64,kernel_size=(5,5),padding="same",activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.15),

    

    # layer 4

    #tf.keras.layers.Conv2D(f*128,kernel_size=(3,3),padding="same",activation='relu'),

    #tf.keras.layers.Conv2D(f*128,kernel_size=(3,3),padding="same",activation='relu'),

    #tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'), #512

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])



print(model.summary())
# model compiling

initial_learningrate=0.001#*0.3

model.compile(optimizer=

              #Adam(learning_rate=0.0003),

              RMSprop(lr=initial_learningrate),

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])
# Set a learning rate annealer

lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='acc', 

                                            patience=300, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
def lr_decay(epoch, initial_learningrate = 0.001):#lrv 0.0003

    return initial_learningrate * 0.99 ** epoch
train_datagen = ImageDataGenerator(#rescale=1./255.,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)





#datagen.fit(X_train)



valid_datagen = ImageDataGenerator(#rescale=1./255.,

                                    horizontal_flip=False,

                                    rotation_range=15,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.15,

                                   zoom_range=0.25,

                                    )



# add early stopping

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)



# fit model with generated data

batchsize = 512*4

epoch = 45



history = model.fit_generator(train_datagen.flow(X, Y, batch_size = batchsize),

                   steps_per_epoch = 100, 

                    epochs = epoch,

                   callbacks=[callback,

                            LearningRateScheduler(lr_decay),

                            lr

                             ],

                   validation_data=valid_datagen.flow(X_dev, Y_dev),

                    validation_steps=50,

                   )
# train model

#history = model.fit(X_train, Y_train, epochs=30,

#          callbacks=[callback],

#          validation_data=[X_dev,Y_dev]

#         )



# evaluate model performance

test_loss, test_acc = model.evaluate(X_dev, Y_dev,verbose=2)



print('\nTest accuracy: ', test_acc)

print('\nTest loss: ', test_loss)
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])



plt.subplot(1,2,2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])
yhat = model.predict_classes(X_valid)

submission['label']=pd.Series(yhat)

submission.to_csv('submission.csv',index=False)
submission.head()
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import h5py



from keras.models import load_model



model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'