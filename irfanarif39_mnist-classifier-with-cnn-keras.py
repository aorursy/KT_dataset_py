# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Taken reference from Kaggle - Deep Learning tutorial and Coursera - Deep Learning Specialization(Andrew Ng), 

# https://www.kaggle.com/ - Kernel mnist-with-keras-for-beginners-99457

# Using CNN

from tensorflow import keras
#Load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

#Convert into array

x_train = (train_data.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train_data.iloc[:,0].values.astype('int32') # only labels

x_test = test_data.values.astype('float32')
#Normalize

x_train = x_train/255.0

x_test = x_test/255.0

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)
#Reshape

X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#Import models, layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization

#from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

batch_size = 64

num_classes = 10

epochs = 25

input_shape = (28, 28, 1)
# convert class vectors to binary class matrices - One Hot Encoding

y_train = keras.utils.to_categorical(y_train, num_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
#Model generation

model = Sequential()

model.add(Conv2D(32, kernel_size= 3,activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(Conv2D(32, kernel_size= 3,activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.20)) #0.20

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Conv2D(64, kernel_size = (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25)) #0.25

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25)) #0.25

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25)) #0.25

model.add(Dense(num_classes, activation='softmax'))
#Compile model

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer= keras.optimizers.RMSprop(),

              metrics=['accuracy'])
#learning Rate reduction

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)
#Data Augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,#15 # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
#model.summary()
#Model train

datagen.fit(X_train)

h = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size)

                       , callbacks=[learning_rate_reduction],)
#Model Evaluate

final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print(final_loss, final_acc)
#get the predictions for the test data

predicted_classes = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("final_submission_V5.csv", index=False, header=True)
"""#Image structure

img_rows, img_cols = 28, 28

#Output values

num_classes = 10



#Prepare data

def prepare_data(rawdata):

    y = rawdata[:,0] # label is present at first column

    #convert output value to categorical values.

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = rawdata[:, 1:] #Take all data except label column

    num_images = rawdata.shape[0]

    #reshaping the data to 1D

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    

    #normalize in range[0,1] for converging CNN

    out_x = out_x/255

    return out_x, out_y



x, y = prepare_data(train_data)

"""
"""#Build CNN model - 3 layers

batch_size = 32

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation = "relu", input_shape = (img_rows, img_cols, 1)))

#model.add(Dropout(0.5))

BatchNormalization(axis=1)

model.add(Conv2D(32, kernel_size = 3, activation = "relu"))

#model.add(Dropout(0.5))

model.add(MaxPool2D())

BatchNormalization(axis=1)

model.add(Conv2D(64, kernel_size = 3, activation = "relu"))

#model.add(Dropout(0.5))

model.add(MaxPool2D())

BatchNormalization(axis=1)

model.add(Conv2D(64, kernel_size = 3, activation = "relu"))

#model.add(Dropout(0.5))

model.add(MaxPool2D())

model.add(Flatten())

BatchNormalization()

model.add(Dense(128, activation = "relu"))

BatchNormalization()

model.add(Dense(num_classes, activation = "softmax"))"""
#Compile model

#model.compile(loss= keras.losses.categorical_crossentropy, optimizer= "sgd", metrics=["accuracy"])

#model.compile(loss= keras.losses.categorical_crossentropy, optimizer= "adadelta", metrics=["accuracy"])

#model.compile(loss= keras.losses.categorical_crossentropy, optimizer= "RMSProp", metrics=["accuracy"])

#model.compile(loss= keras.losses.categorical_crossentropy, optimizer= "adam", metrics=["accuracy"])
# Train model

#model.fit(x, y, batch_size= batch_size, epochs=5, validation_split= 0.2)
# Augmentation technique

#from sklearn.model_selection import train_test_split

#from keras.preprocessing.image import ImageDataGenerator
"""x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.10, random_state=30)





train_gen = ImageDataGenerator(

        rotation_range=10,

        width_shift_range=0.3,

        height_shift_range=0.3,

        shear_range=0.3,

        zoom_range=0.3)



model.fit_generator(train_gen.flow(x = x_train,y = y_train, batch_size=16),epochs=5,

                    steps_per_epoch=X_train.shape[0]//16, )



"""
"""loss, accuracy = model.evaluate(x_val, y_val)

print(loss)

print(accuracy)"""
#Load Test data

#test_data =pd.read_csv("../input/test.csv").values

#test_data[:5,:5]
#test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)

#test_data = test_data/255
#print(test_data.shape)

#print(len(test_data))

#count = len(test_data)
"""# predict results

results = model.predict(test_data)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")"""
"""submission = pd.concat([pd.Series(range(1,(count+1)),name = "ImageId"),results],axis = 1)



submission.to_csv("final_submission.csv",index=False)"""