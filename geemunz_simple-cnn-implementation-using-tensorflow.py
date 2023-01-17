import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

import random

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(tf.__version__)

tf.keras.backend.clear_session()



traindata = pd.read_csv("../input/digit-recognizer/train.csv")

testdata = pd.read_csv("../input/digit-recognizer/test.csv")



print(traindata.isnull().any().describe()) #checking for any null values in the dataset

print(testdata.isnull().any().describe())

print("")

print('train labels', traindata['label'].unique()) # print out unique data labels

print("")

print(traindata['label'].value_counts()) 

print("Train size:{}\nTest size:{}".format(traindata.shape, testdata.shape))

traindata[:5]
labels = traindata.iloc[:,0].values.astype('int32') #extract the labels from the training set

x_train = traindata.iloc[:,1:].values.astype('int32') #extract the 784 pixel values from the training set

x_test = testdata.values.astype('int32') #convert from a pandas df to a numpy array



x_train = x_train.reshape(-1,28,28,1) #reshapeing the data into 28 x 28 pixels

x_test = x_test.reshape(-1,28,28,1)



y_train = tf.keras.utils.to_categorical(labels) #1 hot encode the labels



print(x_train.shape)

print(x_test.shape)

print(y_train)
random_img=random.sample(range(x_train.shape[0]),20)

plt.figure(figsize=(15,9))

for i in random_img:

    plt.subplot(5,10,random_img.index(i)+1)

    plt.title(y_train[i])

    plt.imshow(x_train[i][:,:,0], cmap=cm.binary)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9) #I have set the training size to be 90% of the dataset but this is a value you can play around with.

print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)

image_generator = ImageDataGenerator(

        rescale = 1/255.0, #Normalize the image grayscale value to 0 - 1

        rotation_range = 20, #

        width_shift_range = 0.1, #

        height_shift_range = 0.1, #

        zoom_range = 0.1) #



validation_generator = ImageDataGenerator(

        rescale = 1/255.0)
for X_batch, y_batch in image_generator.flow(x_train, y_train, batch_size=9):

# grid of 3x3 images

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))



    plt.show()

    break
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu')) #The number of hidden nodes can be varied

model.add(tf.keras.layers.Dropout(0.25)) #vary dropout to prevent overfitting

model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='Adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit_generator(image_generator.flow(x_train, y_train,batch_size = 512),

                              epochs =100,validation_data = validation_generator.flow(x_val, y_val, batch_size = 512), verbose=1) #Try different epochs and batch size
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='lower right')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()
predict_generator = ImageDataGenerator(

        rescale = 1/255.0)



predictions = model.predict_generator(predict_generator.flow(x_test, shuffle = False), verbose=0) #Shuffle is set to True by default make sure you set it to False or it will shuffle your test set and cause trouble when submitting to kaggle

predictions = np.argmax(predictions,axis=1)

print(predictions)

pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions}).to_csv("preds.csv", index=False, header=True)