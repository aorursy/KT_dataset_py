# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#load the libs

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import os

print(tf.__version__)

print(os.listdir("../input"))

print("test")
#import data and define the classes

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

class_names = [0,1,2,3,4,5,6,7,8,9]



#print out training data

print(train_data.shape)

print(train_data.head())
from sklearn.model_selection import train_test_split



#split out the data into features (pixel values) and categorical labels (digit values 0-9)

train_x = train_data.iloc[:,1:].values.astype('float32') # all pixel values

train_y = train_data.iloc[:,0].values.astype('int32') # only labels i.e targets digits



test_x = test_data.iloc[:,].values.astype('float32') # all pixel values



#reshape the features to be 28x28

train_x = train_x.reshape(train_x.shape[:1] + (28, 28, 1))

test_x = test_x.reshape(test_x.shape[:1] + (28, 28, 1))



#change the labels to be one-hot encoded

train_y = keras.utils.to_categorical(train_y)

num_classes = train_y.shape[1]





#normalize pixel values using minmax (values between 0 and 1 inclusive)

train_x = train_x / 255

test_x = test_x / 255
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
plt.figure()

plt.imshow(train_x[0].reshape(28, 28))

plt.colorbar()

plt.grid(False)

plt.show()



#plot a group of features and labels to check data

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_x[i].reshape(28, 28), cmap=plt.cm.binary)

    plt.xlabel(class_names[np.argmax(train_y[i])])

plt.show()
from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization



#define the model and layers



#first layer

layer1= tf.keras.layers.Conv2D(32,kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',

                          input_shape=(28,28,1))

layer2= tf.keras.layers.Conv2D(32,kernel_size=(3,3), activation='relu',kernel_initializer='he_normal')

layer3= tf.keras.layers.MaxPooling2D(pool_size=(2,2))

layer4= tf.keras.layers.Dropout(0.20)



#second layer

layer5= tf.keras.layers.Conv2D(64,(3, 3),activation='relu',padding='same')

layer6= tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')

layer7= tf.keras.layers.MaxPooling2D(pool_size=(2,2))

layer8= tf.keras.layers.Dropout(0.25)







## third layer

layer9= tf.keras.layers.Conv2D(128,(3, 3),activation='relu',padding='same')

layer10= tf.keras.layers.Dropout(0.25)



#output layer

layer11= tf.keras.layers.Flatten()

layer12= tf.keras.layers.Dense(128,activation='relu')

# layer12 = tf.keras.layers.BatchNormalization()

layer13= tf.keras.layers.Dropout(0.3)

layer14= tf.keras.layers.Dense(10, activation=tf.nn.softmax)

model = keras.models.Sequential()

model.add(layer1)

model.add(layer2)

model.add(layer3)

model.add(layer4)

model.add(layer5)

model.add(layer6)

model.add(layer7)

model.add(layer8)

model.add(layer9)

model.add(layer10)

model.add(layer11)

# model.add(layer12)

model.add(layer13)

model.add(layer14)
model.summary()
## save model to image



from keras.utils import plot_model

plot_model(model, to_file='model.png')
#compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



#print a summary of the model

#train the model

hist = model.fit(x=train_x, 

            y=train_y,

            batch_size=128,

            epochs=20,

            verbose=1,

            validation_split=0.15,

            shuffle=True)

test = layer1.get_weights()

test[0].shape
plt.imshow(train_x[10][:,:,0])
from numpy import array

import matplotlib.pyplot as plt



def plot_conv_weights():

    W = layer1.get_weights()[0]

    if len(W.shape) == 4:

        W = np.squeeze(W)

        for i in range(W.shape[0]):

            print(i)

            for j in range(W.shape[1]):

                print(j)

                a = array(W[i][j])

                b = a.reshape(4,8)

                plt.imshow(b, cmap='magma', interpolation='nearest')

                name = str(i) + str(j) + ".png"

                plt.savefig(name)

        



            

plot_conv_weights()
lay.shape[0]
#make predictions on the test features

predictions = model.predict(test_x)
def plot_value_array(i, predictions_array):

    predictions_array = predictions_array[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    plt.ylim([0, 1]) 

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')



def plot_image(i, predictions_array, img):

    img = img.reshape(img.shape[0] ,28, 28)

    predictions_array, img = predictions_array[i], img[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} - prob:{:2.0f}%".format(class_names[predicted_label], 100*np.max(predictions_array)), color='red')



# Plot the first X test images, their predicted label, and the true label

# Color correct predictions in blue, incorrect predictions in red

num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_image(i, predictions, test_x)

    plt.subplot(num_rows, 2*num_cols, 2*i+2)

    plot_value_array(i, predictions)

plt.show()
print(hist.history['loss'])

print(hist.history['acc'])

print(hist.history['val_loss'])

print(hist.history['val_acc'])


import matplotlib.pyplot as plt



fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()



loss_ax.plot(hist.history['loss'], 'y', label='train loss')

loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

loss_ax.legend(loc='upper right')



acc_ax.plot(hist.history['acc'], 'b', label='train acc')

acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

acc_ax.set_ylabel('accuracy')

acc_ax.legend(loc='upper left')



plt.show()
#submissions for Kaggle

#cat_predictions = np.argmax(predictions, axis=1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": np.argmax(predictions, axis=1)})

submissions.to_csv("my_submissions.csv", index=False, header=True)