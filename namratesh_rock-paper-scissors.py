# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

        print(os.path.join(dirname))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as  np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



#defining our training and validation data set path

train_data = "/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/Rock-Paper-Scissors/train"

validation_data = "/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/Rock-Paper-Scissors/test"
#printing images



def read_img(path):

    img=mpimg.imread(path)

    imgplot = plt.imshow(img)

    return plt.show()



read_img("/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/Rock-Paper-Scissors/train/rock/rock01-001.png")



read_img("/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/Rock-Paper-Scissors/train/paper/paper01-005.png")



read_img("/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/Rock-Paper-Scissors/train/scissors/scissors01-005.png")

    
#definig a callback to stop our model when it gets 95%

class myCallbacks(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc')>0.95):

            print("\nReached 92 accuracy so cancelling training!")

            self.model.stop_training = True

    
#importing dataset 

train_genr = ImageDataGenerator(rescale = 1./255.)

validation_genr = ImageDataGenerator(rescale = 1./255.)



train_imgs = train_genr.flow_from_directory(directory = train_data, batch_size = 32,

                                                  target_size = (300,300), class_mode = 'categorical' )



val_imgs = validation_genr.flow_from_directory(directory = validation_data, batch_size = 32,

                                                  target_size = (300,300), class_mode = 'categorical' )

#initiallising call backs

callbacks = myCallbacks()
model = tf.keras.models.Sequential([

    #adding convolution layer and maxpooling layers

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3*3), activation = 'relu', input_shape = (300,300,3)),

    tf.keras.layers.MaxPooling2D(pool_size = (2*2)),

    tf.keras.layers.Conv2D(64, kernel_size = (3*3), padding = 'same', activation = 'relu'),

    tf.keras.layers.MaxPooling2D(pool_size = (2*2)),

    tf.keras.layers.Conv2D(128, kernel_size = (3*3), padding = 'same', activation = 'relu'),

    tf.keras.layers.MaxPooling2D(pool_size = (2*2)),

    

    #flatting  array because dense layer takes input in 1d

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation = 'relu'),

    tf.keras.layers.Dense(3, activation = 'softmax')



])







#compiling the model

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])



#summary of model

model.summary()
#fitting the model

history = model.fit(train_imgs, validation_data = val_imgs, epochs = 20, steps_per_epoch = 30, verbose = 1, callbacks = [callbacks])
import matplotlib.pyplot as plt

acc= history.history['acc']

loss = history.history['loss']



val_loss = history.history['val_loss']

val_acc = history.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r' ,label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', label = 'validation Accuracy')

plt.title("Training and validation Accuracy")

plt.legend()

plt.figure()







plt.plot(epochs, loss, 'r',  label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'validation loss')

plt.title("Training and validation loss")

plt.legend()

plt.figure()
model.save('h.h5') #save our model for further use
from tensorflow.keras.models import load_model

model  = load_model('h.h5')

model.summary()
import numpy as np

from keras.preprocessing import image

path = '/kaggle/input/rock-paper-scissors-dataset/rock-paper-scissors/Rock-Paper-Scissors/validation/paper5.png'

read_img(path)

#reading images

img = image.load_img(path, target_size = (300,300))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

images = np.expand_dims(x, axis = 0)

images = np.vstack([x])

def predict_img(img):

    pred_clss = model.predict_classes(images)

    print("prediction class is :", pred_clss)

    if pred_clss ==array([1]):

        print('rock')

    elif pred_clss ==array([0]):

        print('paper')

    if pred_clss ==array([1]):

        print('scissor')

    

          
predict_img(path)