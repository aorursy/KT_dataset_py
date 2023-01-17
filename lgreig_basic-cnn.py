#Load the files 

import numpy as np 

import pandas as pd 





train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



target = train['label']



train  = train.drop(['label'], axis=1)

#Convert the train and test to arrays and reshape and normalize

full = pd.concat([train, test])

full=full.to_numpy()

full=full.reshape(-1, 28, 28, 1)

full=full.astype("float32")

full = full/255

train=full[:42000, :, :, :]

test=full[42000:, :, :, :]
#Convert the target to a dummy variable 

import tensorflow as tf

from keras.utils import to_categorical

target=to_categorical(target, num_classes=10)
#Split data for training

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(train, target, test_size = 0.2, random_state = 1 )
#Required libraries for model

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt
#Define model as function

def NN_model():

    

    NN = Sequential()

    NN.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))

    NN.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))

    NN.add(MaxPooling2D((2, 2)))

    NN.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))

    NN.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))

    NN.add(MaxPooling2D((2, 2)))

    NN.add(Flatten())

    NN.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

    NN.add(Dense(10, activation='softmax'))

    #compile model

    opt = SGD(lr=0.01, momentum=0.9, decay=0.0005)

    NN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return NN

fig = plt.figure()



for idx in range(8):

    ax = fig.add_subplot(3, 3, idx+1)

    ax.imshow(train[idx, :,:, 0], cmap="gray")
#Create an image generator class for augmentation to improve generalisation

from keras.preprocessing.image import ImageDataGenerator

im = ImageDataGenerator(zoom_range=0.1, rotation_range=15, height_shift_range= 0.05, width_shift_range= 0.05)

#im = ImageDataGenerator()

flow=im.flow(X_train, Y_train, batch_size=32)



fig = plt.figure()

for i in range(9):

    ax = fig.add_subplot(3,3, i+1)

    batch = flow.next()

    image = batch[0][0][:, :,0]

    ax.imshow(image)

mymod=NN_model()

perf = mymod.fit_generator(flow, epochs=12, steps_per_epoch= X_train.shape[0]/32, validation_data=(X_val, Y_val), verbose=1)
plt.plot(perf.history["val_accuracy"])

plt.show()
preds = mymod.predict(test)

preds = np.argmax(preds,axis = 1)

preds= pd.Series(preds,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),preds],axis = 1)

submission.to_csv("MINST.csv",index=False)