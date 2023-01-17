import numpy as np

import h5py



from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.models import Sequential

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model





import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline
def load_dataset():

    train_data = h5py.File('../input/train_happy.h5', "r")

    X_train = np.array(train_data["train_set_x"][:]) 

    y_train = np.array(train_data["train_set_y"][:]) 



    test_data = h5py.File('../input/test_happy.h5', "r")

    X_test = np.array(test_data["test_set_x"][:])

    y_test = np.array(test_data["test_set_y"][:]) 

    

    y_train = y_train.reshape((1, y_train.shape[0]))

    y_test = y_test.reshape((1, y_test.shape[0]))

    

    return X_train, y_train, X_test, y_test
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()



# Normalize image vectors

X_train = X_train_orig/255.

X_test = X_test_orig/255.



# Reshape

Y_train = Y_train_orig.T

Y_test = Y_test_orig.T



print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))
# Sample image from dataset

print("Image shape :",X_train_orig[10].shape)

imshow(X_train_orig[10])
# GRADED FUNCTION: HappyModel



def HappyModel(input_shape):

    

    X_input = Input(input_shape)

    

    X = ZeroPadding2D((3,3))(X_input)

    

    # CONV -> BN -> RELU Block applied to X

    X = Conv2D(32, (7,7), strides=(1,1), name='Conv2D')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D((2,2), name='max_pool')(X)

    

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)

    

    model = Model(inputs = X_input, outputs=X, name='HappyModel')

        

    return model
# Model flow chart

happyModel = HappyModel(X_train[0].shape)

plot_model(happyModel, to_file='HappyModel.png')

SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
happyModel.summary()
happyModel_sgd = HappyModel(X_train.shape[1:])

happyModel_sgd.compile(optimizer='sgd', loss='binary_crossentropy', metrics=["accuracy"])
history_sgd = happyModel_sgd.fit(X_train,Y_train, epochs=5,batch_size=30)
train_accuracy = history_sgd.history['acc']

train_loss = history_sgd.history['loss']



iterations = range(len(train_accuracy))

plt.plot(iterations, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(iterations, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds = happyModel_sgd.evaluate(x=X_test, y=Y_test)



print ("\nLoss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
happyModel_rms = HappyModel(X_train.shape[1:])

happyModel_rms.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=["accuracy"])
history_rms = happyModel_rms.fit(X_train,Y_train, epochs=5,batch_size=30)
train_accuracy = history_rms.history['acc']

train_loss = history_rms.history['loss']



iterations = range(len(train_accuracy))

plt.plot(iterations, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(iterations, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds = happyModel_rms.evaluate(x=X_test, y=Y_test)



print ("\nLoss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
happyModel_adam = HappyModel(X_train[0].shape)

happyModel_adam.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
history_adam = happyModel_adam.fit(X_train,Y_train, epochs=5,batch_size=30)
train_accuracy = history_adam.history['acc']

train_loss = history_adam.history['loss']



count = range(len(train_accuracy))

plt.plot(count, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(count, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds = happyModel_adam.evaluate(x=X_test, y=Y_test)



print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
happyModelE = HappyModel(X_train.shape[1:])

happyModelE.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
THistoryE = happyModelE.fit(X_train,Y_train, epochs=20,batch_size=30)
train_accuracy = THistoryE.history['acc']

train_loss = THistoryE.history['loss']



iterations = range(len(train_accuracy))

plt.plot(iterations, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(iterations, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds = happyModelE.evaluate(x=X_test, y=Y_test)



print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
y_pred = happyModelE.predict(X_test)
y_pred[y_pred < 0.5] = 0

y_pred[y_pred >= 0.5] = 1
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred)

print(cm)
happyModel3 = HappyModel(X_train.shape[1:])

happyModel3.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
# THistory (Train History)

THistory3 = happyModel3.fit(X_train,Y_train, epochs=30,batch_size=30)
train_accuracy = THistory3.history['acc']

train_loss = THistory3.history['loss']



iterations = range(len(train_accuracy))

plt.plot(iterations, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(iterations, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds = happyModel3.evaluate(x=X_test, y=Y_test)



print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
happyModel2 = HappyModel(X_train[0].shape)

happyModel2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
THistory2 = happyModel2.fit(X_train,Y_train, epochs=40,batch_size=16)
train_accuracy = THistory2.history['acc']

train_loss = THistory2.history['loss']



iterations = range(len(train_accuracy))

plt.plot(iterations, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(iterations, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds2 = happyModel2.evaluate(x=X_test, y=Y_test)



print()

print ("Loss = " + str(preds2[0]))

print ("Test Accuracy = " + str(preds2[1]))
# Building LeNet-5 

def create_model():

    model = Sequential()

    model.add(layers.Conv2D(filters=1, kernel_size=(1,1), strides=(2,2), name='Conv2D', input_shape=(64,64,3))) # For converting image to 32,32,1

    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))

    model.add(layers.AveragePooling2D())



    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))

    model.add(layers.AveragePooling2D())



    model.add(layers.Flatten())



    model.add(layers.Dense(units=120, activation='relu'))



    model.add(layers.Dense(units=84, activation='relu'))



    model.add(layers.Dense(units=1, activation = 'sigmoid'))

    

    return model
model = create_model()

model.summary()
plot_model(model, to_file='HappyModel.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
lenet5 = create_model()

lenet5.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])

history = lenet5.fit(X_train,Y_train, epochs=20,batch_size=32)



train_accuracy = history.history['acc']

train_loss = history.history['loss']



iterations = range(len(train_accuracy))

plt.plot(iterations, train_accuracy, label='Training accuracy')

plt.title('epochs vs Training accuracy')

plt.legend()



plt.figure()

plt.plot(iterations, train_loss, label='Training Loss')

plt.title('epochs vs Training Loss')

plt.legend()
preds = lenet5.evaluate(x=X_test, y=Y_test)



print ("\nLoss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
