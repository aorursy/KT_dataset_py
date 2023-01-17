from IPython.display import Image

Image("../input/amer_sign3.png")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import tensorflow as tf

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, SpatialDropout2D, Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential



import matplotlib.pyplot as plt
# set up training data and labels

dim_x = 28

dim_y = 28

batch_size=32



# read in data/labels

x_train = pd.read_csv('../input/sign_mnist_train.csv')

x_train.head()

x_train.shape

y_train = np.array(x_train['label'])

x_train.drop('label', axis = 1, inplace = True)

x_train = np.array(x_train.values)



print("data shapes", x_train.shape, y_train.shape, "classes: ",len(np.unique(y_train)))



classes = len(np.unique(y_train))

x_train = x_train.reshape((-1, dim_x,dim_y,1))

# convert labels to one-hot

print(np.unique(y_train))

y = np.zeros((np.shape(y_train)[0],len(np.unique(y_train))))



# skip over 'j'

y_train[y_train>8] = y_train[y_train>8] - 1



# convert index labels to one-hot

for ii in range(len(y_train)):

    #print(y_train[ii])

    y[ii,y_train[ii]] = 1

y_train = y
# split into training/validation

no_validation = int(0.1 * (x_train.shape[0]))



x_val = x_train[0:no_validation,...]

y_val = y_train[0:no_validation,...]



x_train = x_train[no_validation:,...]

y_train = y_train[no_validation:,...]



print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)



# define image generators with mild augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,\

                                   rotation_range=10,\

                                   width_shift_range=0.05,\

                                   height_shift_range=0.05,\

                                   shear_range=0.1,\

                                   zoom_range=0.075)



train_generator = train_datagen.flow(x=x_train,\

                                     y=y_train,\

                                     batch_size=batch_size,\

                                     shuffle=True)



test_datagen = ImageDataGenerator(rescale=1./255)



val_generator = test_datagen.flow(x=x_val,\

                                    y=y_val,\

                                    batch_size=batch_size,\

                                    shuffle=True)

# define model Le-Net5

model = Sequential()



model.add(Conv2D(filters=6, kernel_size=(5,5), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.tanh))

model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=16, kernel_size=(5,5), strides=1, activation=tf.nn.tanh))

model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

model.add(Dense(120,activation=tf.nn.tanh))

model.add(Dense(84,activation=tf.nn.tanh))

model.add(Dense(classes, activation=tf.nn.softmax))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
steps_per_epoch = int(len(y_train)/batch_size)

max_epochs = 64

history = model.fit_generator(generator=train_generator,\

                                steps_per_epoch=steps_per_epoch,\

                                validation_data=val_generator,\

                                validation_steps=50,\

                                epochs=max_epochs,\

                                verbose=2)
plt.figure(figsize=(15,8))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Accuracy",fontsize=28)

plt.xlabel('epoch',fontsize=18)

plt.ylabel('accuracy',fontsize=18)

plt.legend(['Train','Val'],fontsize=18)

plt.show()
x_test = pd.read_csv('../input/sign_mnist_test.csv')

y_test = x_test['label']

x_test.head()


x_test.drop('label', axis = 1, inplace = True)

x_test = np.array(x_test.values)

x_test = x_test / 255.



print("data shape", x_test.shape)



x_test = x_test.reshape((-1, dim_x,dim_y,1))


# convert labels to one-hot

print(np.unique(y_test))



y_temp = np.zeros((np.shape(y_test)[0],len(np.unique(y_test))))



y_test[y_test>8] = y_test[y_test>8] - 1



for ii in range(len(y_test)):

    #print(y_train[ii])

    y_temp[ii,y_test[ii]] = 1

y_test = y_temp
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred.round())
# see how we did



def imshow_w_labels(img, target, pred):

    plt.figure(figsize=(6,6))

    plt.imshow(img, cmap="gray")

    plt.title("Prediction: %s, Target: %s"%(target,pred), fontsize=24)

    plt.show()



letters = {}

counter = 0

for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']:

    letters[counter] = letter

    counter += 1

    

for kk in range(50,60):

    imshow_w_labels(x_test[kk,:,:,0],letters[y_test[kk].argmax()], letters[y_pred[kk].argmax()])