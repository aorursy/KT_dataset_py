# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np # forlinear algebra

import matplotlib.pyplot as plt #for plotting things

import os

from PIL import Image

print(os.listdir("../input"))





import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img
import os

mainDIR = os.listdir('../input/chest-xray-pneumonia/chest_xray')

print(mainDIR)
train_folder=  "../input/chest-xray-pneumonia/chest_xray/train/"

val_folder =  os.listdir('../input/chest-xray-pneumonia/chest_xray/val/')

test_folder =  os.listdir('../input/chest-xray-pneumonia/chest_xray/test/')



# train 

os.listdir(train_folder)

train_n = train_folder+'NORMAL/'

train_p = train_folder+'PNEUMONIA/'

#Normal pic 

print(len(os.listdir(train_n))) #will print the total no of files in the directory

rand_norm= np.random.randint(0,len(os.listdir(train_n))) #random sampling returns an array of shape.

norm_pic = os.listdir(train_n)[rand_norm]

print('normal picture title: ',norm_pic)

norm_pic_address = train_n+norm_pic
#Pneumonia

rand_p = np.random.randint(0,len(os.listdir(train_p)))



sic_pic =  os.listdir(train_p)[rand_norm]

sic_address = train_p+sic_pic

print('pneumonia picture title:', sic_pic)
# Load the images

norm_load = Image.open(norm_pic_address)

sic_load = Image.open(sic_address)
 #split these images

f = plt.figure(figsize= (10,6)) #10 width and 6 height

a1 = f.add_subplot(1,2,1) #(height, width, plot number.)as paramenter divides fig into objects

img_plot = plt.imshow(norm_load)  #imshow() creates an image from a 2-dimensional numpy array

a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title('Pneumonia')
# the CNN model



cnn = Sequential()

#add() to add layers to cnn.

#Convolution

cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))



#Pooling

cnn.add(MaxPooling2D(pool_size = (2, 2)))



# 2nd Convolution

cnn.add(Conv2D(32, (3, 3), activation="relu"))



# 2nd Pooling layer

cnn.add(MaxPooling2D(pool_size = (2, 2)))

#Flatten serves as a connection between the convolution and dense layers.

# Flatten the layer

cnn.add(Flatten())



# Fully Connected Layers

cnn.add(Dense(activation = 'relu', units = 128))

cnn.add(Dense(activation = 'sigmoid', units = 1))

#The optimizer controls the learning rate. 

#The learning rate determines how fast the optimal weights for the model are calculated.

# Compile the Neural network

# ???accuracy??? metric to see the accuracy score on the validation set when we train the model.

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

# The function ImageDataGenerator augments your image by iterating through image as your CNN is 

#getting ready to process that image



train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.



training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train/',target_size = (64, 64),batch_size = 32,class_mode = 'binary')



validation_generator = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val/',target_size=(64, 64),batch_size=32,class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test/',target_size = (64, 64), batch_size = 32,class_mode = 'binary')

cnn.summary()
cnn_model = cnn.fit_generator(training_set,steps_per_epoch = 100,epochs = 10, validation_data = validation_generator,validation_steps = 20)
test_accu = cnn.evaluate_generator(test_set,steps=100)

acc=test_accu[1]*10

print('The testing accuracy is :', test_accu[1]*100 ,'%')

# Accuracy 

plt.plot(cnn_model.history['accuracy'])

plt.plot(cnn_model.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()

# Loss 



plt.plot(cnn_model.history['val_loss'])

plt.plot(cnn_model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()