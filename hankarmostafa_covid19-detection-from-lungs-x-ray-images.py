import glob
import numpy as np 
import cv2
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import tensorflow as tf 
%matplotlib inline 
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



DIR = "../input/covid-19-x-ray-10000-images/dataset"
os.listdir(DIR)

# Gather Normal Images 
normal_images = []
for img_path in glob.glob(DIR + '/normal/*'):
    normal_images.append(cv2.imread(img_path))

fig = plt.figure()
fig.suptitle('Normal X- Ray Image')
plt.imshow(normal_images[0], cmap='gray') 
# GRAB cOVID iMAGES 
covid_images = []
for img_path in glob.glob(DIR + '/covid/*'):
    covid_images.append(cv2.imread(img_path))

fig = plt.figure()
fig.suptitle('Covid X-Ray Image')
plt.imshow(covid_images[0], cmap='gray') 
len(covid_images)
# show the lenghth of the two lists of images 
print(f"Number of Normal Images :{format(len(normal_images))}")
print(f"Number of Covid Images  :{format(len(covid_images))}")
W = 150 # Width of Images 
H= 150  # height of Images 
C= 3    # Number of Channels . 3 for RGB 

input_shape= (W, H, C)
N_CLASSES = 2
EPOCHS = 20
BATCH_SIZE = 5
## The Neural network archiecture will be like this :
 ##1 convlution layer
 ##2 relu layer
 ##3 pooling layer
 ##4 fully connected

model = Sequential()
# First Layer 
model.add(Conv2D(filters = 64, kernel_size = (3,3) ,activation ='relu', 
                 input_shape = input_shape))
model.add(Conv2D(filters = 56, kernel_size = (3,3),activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Second alyer 
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 48, kernel_size = (3,3),activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Third Layer 
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
# show model summary 
model.summary()
data_generator = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest', # Fill in missing pixels with the nearest filled value
                               validation_split=0.25)

train_generator= data_generator.flow_from_directory(
                        DIR,
                        target_size=(H, W),
                       
                        class_mode='binary',
                        subset='training')

test_generator= data_generator.flow_from_directory(
    DIR, 
    target_size=(H,W),
    class_mode='binary',
    shuffle= False,
    subset='validation')

# Train our Model 
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = test_generator, 
    validation_steps = test_generator.samples // BATCH_SIZE,
    epochs = 10 )
history.params['metrics']
# Plot training loss vs validation loss 
plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('loss')
ax1.set_xlabel('epochs')
## plot training accuracy vs validation accuracy 
ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epochs')
preds= model.predict(test_generator)
predicted_class_indices=np.argmax(preds,axis=1)
labels = (test_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print (labels)
print (predictions)
# put them on a Serie List 
predicted_values = pd.Series(predictions)
# create a submission dataframe 
submission = pd.DataFrame({'Image Id':predicted_values.index,'Status':predicted_values })
submission.set_index('Image Id',inplace=True)
# save to a csv file 
submission.to_csv('covid19_xray_submission.csv', index=False)
print(" Submission  successfully saved!")
submission.head()
