# get access to the google drive  data

from google.colab import drive
drive.mount('/content/drive')
#allo to get access drive's id
from google.colab import drive
drive.mount('/content/drive')
!unrar x "/content/drive/My Drive/corona dataset.rar" 
!pip install -q keras
#importing models in the keras +layers for extracting data
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten ,Dense

#Initalization of model of prediction of corona virus
classifier=Sequential()  

#1nd conv layer
#step 1 conv layer
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#step 2 Polling  layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2nd conv layer
#step 1 conv layer
classifier.add(Conv2D(16,3,3,activation='relu'))
#step 2 Polling  layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
#sttep 3 Flatten layer
classifier.add(Flatten())

#step 4 -Full Connection
classifier.add(Dense(units=128,activation='relu'))
#adding  1 hidden layers in the  NET of FULLY CONNECTION
classifier.add(Dense(units=1,activation='sigmoid'))  # for prediction 1 OR 0    else we can use softmax for better predcition
#adding  2 hidden layers in the  NET of FULLY CONNECTION

#compile //optimizer ADAm/RMS PROP                loss(MSE/cross entropy)     metrics accuarcy
classifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/corona dataset/train_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/content/corona dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 700,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 200)
#final our model get trained to predict CAT OR DOG
#prediction whelether image is of dog or cat
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('/content/corona dataset/single_prediction/5%2.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
  print("The Person is suffering from covid.")
else:
  print("The Person is not suffering from covid.")

print("MODEL COMPLETED PREDICT THE RIGHT IMAGE .")
#prediction 
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('/content/corona dataset/single_prediction/2020.02.10.20021584-p6-52%7.png',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
  print("The Person is suffering from covid.")
else:
  print("The Person is not suffering from covid.")

print("MODEL COMPLETED PREDICT THE RIGHT IMAGE .")
!pip install ann_visualizer

import keras;
from keras.models import Sequential;
from keras.layers import Dense;
from ann_visualizer.visualize import ann_viz
ann_viz(classifier,  filename='CNN_ARCHITECTURE.gv',title=" CNN VISUALIZER")



