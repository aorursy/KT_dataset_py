# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/chest_xray/chest_xray"))







# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
#preproessing-img aumentation----avoid overfitting

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

#zoom-random

#enough transformations

test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

        '../input/chest_xray/chest_xray/train',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



test_set = test_datagen.flow_from_directory(

        '../input/chest_xray/chest_xray/test',

        target_size=(64,64),

        batch_size=256,

        class_mode='binary')

import random

import os

import matplotlib.pyplot as plt

normal_img = os.listdir('../input/chest_xray/chest_xray/train/NORMAL/')







random = random.sample(normal_img,25)

f,ax = plt.subplots(5,5)



for i in range(0,25):

    im = plt.imread('../input/chest_xray/chest_xray/train/NORMAL/'+random[i])

    ax[i//5,i%5].imshow(im)

    ax[i//5,i%5].axis('off')

f.suptitle('Normal Lungs')

plt.show()



#init creates object of class

classifier=Sequential()

#changes for thaneo

#activation fn-rectifier

classifier.add(Convolution2D(64,3,3,input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))





classifier.add(Flatten())

#full connwection--output either sigmoid or softmax

classifier.add(Dense(output_dim=128,activation='relu'))



classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compile

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



classifier.summary()







history=classifier.fit_generator(training_set,steps_per_epoch=100,epochs=8,validation_data=test_set,validation_steps=50)
%matplotlib inline

import glob

import matplotlib

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import imageio as im

from keras import models

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint


img_path='../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg'

img = image.load_img(img_path,target_size=(64,64))





img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

plt.imshow(img_tensor[0])

plt.show()

print(img_tensor.shape)
from keras.models import load_model

layer_outputs = [layer.output for layer in classifier.layers[:12]] 

# Extracts the outputs of the top 12 layers

activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) 
activations = activation_model.predict(img_tensor) 

# Returns a list of five Numpy arrays: one array per layer activation

first_layer_activation = activations[0]

print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
layer_names = []

for layer in classifier.layers[:4]:

    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    

images_per_row = 10



for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps

    n_features = layer_activation.shape[-1] # Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    

    for col in range(n_cols): # Tiles each filter into a big horizontal grid

        for row in range(images_per_row):

            channel_image = layer_activation[0,:, :,col * images_per_row + row]

            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size, # Displays the grid

                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    

    plt.imshow(display_grid, aspect='auto', cmap='viridis')
from keras.models import load_model



# Creates a HDF5 file 'my_model.h5'

classifier.save('cnn_model.h5')
history
import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from keras.models import load_model



# Creates a HDF5 file 'my_model.h5'

classifier.save('cnn_model.h5')



  

# Returns a compiled model identical to the previous one

classifier = load_model('cnn_model.h5')
import numpy as np

import tensorflow as tf



from keras.preprocessing import image



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



    

model = tf.keras.models.load_model("cnn_model.h5")

test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(64,64))

test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=model.predict(test_image)

training_set.class_indices

if result[0][0]==1:

    prediction='normal'

else:

    prediction='pneumonia'

print(prediction)






predicted_classes = np.argmax(predictions, axis=1)
predicted_classes
