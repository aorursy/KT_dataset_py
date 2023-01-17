# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import tensorflow as tf

from sklearn.model_selection import train_test_split

import keras

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.layers import LeakyReLU

from keras.layers.normalization import BatchNormalization

from tensorflow.keras.callbacks import LearningRateScheduler



import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print("Train set shape = " +str(train.shape))

print("Test set shape = " +str(test.shape))
X=train.iloc[:,1:].values 

Y=train.iloc[:,0].values 
X = X.reshape(X.shape[0], 28, 28,1) 

print(X.shape)

Y = keras.utils.to_categorical(Y, 10) 

print(Y.shape)
x_test=test.iloc[:,:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

x_test.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.15, random_state=42) 
train_datagen = ImageDataGenerator(rescale = 1./255.,

                                   rotation_range = 10,

                                   width_shift_range = 0.15,

                                   height_shift_range = 0.15,

                                   shear_range = 0.1,

                                   zoom_range = 0.2,

                                   horizontal_flip = False)
valid_datagen = ImageDataGenerator(rescale=1./255) 
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(64, (3,3), padding='same'),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),    

    

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.summary()
initial_learningrate=1e-3 

batch_size = 128

epochs = 40

input_shape = (28, 28, 1)
def lr_decay(epoch):#lrv

    return initial_learningrate * 0.9 ** epoch
model.compile(loss="categorical_crossentropy",

              optimizer=RMSprop(lr=initial_learningrate),

              metrics=['accuracy'])
history = model.fit_generator(

      train_datagen.flow(X_train,Y_train, batch_size=batch_size),

      steps_per_epoch=100,

      epochs=epochs,

      callbacks=[LearningRateScheduler(lr_decay) 

               ],

      validation_data=valid_datagen.flow(X_valid,Y_valid),

      validation_steps=50,  

      verbose=2)
accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')

plt.title('Accuracy')

plt.legend()

plt.show()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Test loss')

plt.title('Loss')

plt.legend()

plt.show()
predictions = model.predict_classes(x_test/255.)
final=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})
final.to_csv("cnn_submission.csv",index=False)
