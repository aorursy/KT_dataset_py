# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input,decode_predictions

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam



base_model= MobileNet(weights='imagenet',include_top=False) 



"""

image_path = '../input/data/data/Training/Dates/0_100.jpg'

img = image.load_img(image_path,target_size=(224,224))

x = image.img_to_array(img)

x =np.expand_dims(x,axis =0)

x =preprocess_input(x)

print('input image shape',x.shape)

preds = base_model.predict(x)

print('predicted',decode_predictions(preds))

"""





x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dense(512,activation='relu')(x) #dense layer 3

preds=Dense(3,activation='softmax')(x) #final layer with softmax activation



model=Model(inputs=base_model.input,outputs=preds)

for i,layer in enumerate(model.layers):

  print(i,layer.name)



for layer in model.layers:

    layer.trainable=False

# or if we want to set the first 20 layers of the network to be non-trainable

for layer in model.layers[:20]:

    layer.trainable=False

for layer in model.layers[20:]:

    layer.trainable=True



# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies





#Note: when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample).





test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/data/data/Training',

                target_size = (64, 64),

                batch_size = 32,

                class_mode = 'categorical')





#steps_per_epoch - Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. 

#The default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.

#validation_steps: Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.



import datetime

print("Start time:",datetime.datetime.now())

test_set = test_datagen.flow_from_directory('../input/data/data/Validation',

           target_size = (64, 64),

           batch_size = 32,

           class_mode = 'categorical')





model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])



model.fit_generator(training_set,

                        steps_per_epoch = 10,

                        epochs = 10,

                        validation_data = test_set,

                        validation_steps = 10

                        )

print("End time:",datetime.datetime.now())



from keras.models import load_model



#classifier.save('../input/multiclass_model_latest.h5')  # creates a HDF5 file 'my_model.h5'

#del model  # deletes the existing model



# returns a compiled model

# identical to the previous one

#classifier = load_model('../input/multiclass_model_latest.h5')



# class indices

print(training_set.class_indices)



# test image

import cv2

img = cv2.imread("../input/data/data/Validation/Banana/4_100.jpg")

print("image shape:",img.shape)

img = cv2.resize(img,(64,64))

print("image shape:",img.shape)



best_threshold = [0.4,0.4,0.4]



import numpy as np

img = img.astype('float32')

img = img/255

img = np.expand_dims(img,axis=0)

img.shape

pred = model.predict(img)

print("prediction probabilities:",pred)



y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])



print("ypred",y_pred)



classes = ['Apple Braeburn','Banana','Dates']

output_class = [classes[i] for i in range(3) if y_pred[i]==1 ]  #extracting actual class name

print("Predicted Class is ",output_class[0])



## 2nd test

import cv2

img = cv2.imread("../input/data/data/Validation/Apple Braeburn/1_100.jpg")

print("image shape:",img.shape)

img = cv2.resize(img,(64,64))

print("image shape:",img.shape)



import numpy as np

img = img.astype('float32')

img = img/255

img = np.expand_dims(img,axis=0)

img.shape

pred = model.predict(img)

print("prediction probabilities:",pred)

best_threshold = [0.4,0.4,0.4]

y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])



print("ypred",y_pred)



classes = ['Apple Braeburn','Banana','Dates']

output_class = [classes[i] for i in range(3) if y_pred[i]==1 ]  #extracting actual class name

print("Predicted Class is ",output_class)



## 3rd test

import cv2

img = cv2.imread("../input/data/data/Validation/Dates/13_100.jpg")

print("image shape:",img.shape)

img = cv2.resize(img,(64,64))

print("image shape:",img.shape)



import numpy as np

img = img.astype('float32')

img = img/255

img = np.expand_dims(img,axis=0)

img.shape

pred = model.predict(img)

print("prediction probabilities:",pred)



y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])



print("ypred",y_pred)



classes = ['Apple Braeburn','Banana','Dates']

output_class = [classes[i] for i in range(3) if y_pred[i]==1 ]  #extracting actual class name

print("Predicted Class is ",output_class[0])