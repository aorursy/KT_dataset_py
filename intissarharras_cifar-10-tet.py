#------------------------------------------------------------------------------

# VGG16 ON CIFAR_10

#------------------------------------------------------------------------------

import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow.keras as k

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout

from keras.utils.np_utils import to_categorical

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score





#------------------------------------------------------------------------------

# Using VGG16 model, with weights pre-trained on ImageNet.

#------------------------------------------------------------------------------



vgg16_model = VGG16(weights='imagenet',

                    include_top=False, 

                    classes=10,

                    input_shape=(32,32,3)# input: 32x32 images with 3 channels -> (32, 32, 3) tensors.

                   )



#Define the sequential model and add th VGG's layers to it

model = Sequential()

for layer in vgg16_model.layers:

    model.add(layer)





#------------------------------------------------------------------------------

# Adding hiddens  and output layer to our model

#------------------------------------------------------------------------------



from tensorflow.keras.layers import Dense, Flatten, Dropout

model.add(Flatten())

model.add(Dense(512, activation='relu', name='hidden1'))

model.add(Dropout(0.4))

model.add(Dense(256, activation='relu', name='hidden2'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax', name='predictions'))



model.summary()





#------------------------------------------------------------------------------

#  Loading CIFAR10 data

#------------------------------------------------------------------------------



(X_train, y_train), (X_test, y_test) = k.datasets.cifar10.load_data()



print("******************")

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

 

# Convert class vectors to binary class matrices using one hot encoding

y_train_ohe = to_categorical(y_train, num_classes = 10)

y_test_ohe = to_categorical(y_test, num_classes = 10)



# Data normalization

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train  /= 255

X_test /= 255



print("******************")

print(X_train.shape)

print(y_train_ohe.shape)

print(X_test.shape)

print(y_test_ohe.shape)





#------------------------------------------------------------------------------

# TRAINING THE CNN ON THE TRAIN/VALIDATION DATA

#------------------------------------------------------------------------------



# initiate SGD optimizer

sgd = optimizers.SGD(lr=0.001, momentum=0.9)



# For a multi-class classification problem

model.compile(loss='categorical_crossentropy',optimizer= sgd,metrics=['accuracy'])





es = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')

mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', save_best_only=True, mode='max')





# initialize the number of epochs and batch size

EPOCHS = 100

BS = 32



# construct the training image generator for data augmentation

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

                        horizontal_flip=True, fill_mode="nearest")

 

# train the model

history = model.fit_generator(aug.flow(X_train,y_train_ohe, batch_size=BS),validation_data=(X_test,y_test_ohe), 

                              steps_per_epoch=len(X_train) // BS,epochs=EPOCHS,callbacks=[es,mc])



#We load the best weights saved by the ModelCheckpoint

model.load_weights('./weights.h5')





#------------------------------------------------------------------------------

# PREDICT AND EVALUATE THE CNN ON THE TEST DATA

#------------------------------------------------------------------------------

preds = model.predict(X_test)#on prédit le Test

score_test = accuracy_score( y_test, np.argmax(preds, axis=1) )

print (' LE SCORE DE TEST : ', score_test)





train_loss, train_score = model.evaluate(X_train, y_train_ohe)

test_loss, test_score = model.evaluate(X_test, y_test_ohe)

print("Train Loss:", train_loss)

print("Test Loss:", test_loss)

print("Train Score:", train_score)

print("Test Score:", test_score)