import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.image import imread
data_dir='../input/pedestrian-no-pedestrian/data'
os.listdir(data_dir)
#Set Train and Validation directories

train_path=data_dir+'/train/'

validation_path=data_dir+'/validation/'
validation_path
os.listdir(validation_path)
os.listdir(train_path)
os.listdir(train_path+'pedestrian')[5]
pedestrian=train_path+'pedestrian/'+'pic_073.jpg'

imread(pedestrian).shape
# First two numbers shows the size of the matrix and the number 3 signifies it has combination of 3 colors (Red,Green,Blue)

# Let us visualise a sample image

plt.imshow(imread(pedestrian))
# Now let us see one data from no pedestiran dataset

os.listdir(train_path+'no pedestrian')[5]
no_pedestrian=train_path+'no pedestrian/'+'train (612).jpg'
plt.imshow(imread(no_pedestrian))
# let us check the number of images in dataset

len(os.listdir(train_path+'pedestrian'))
len(os.listdir(train_path+'no pedestrian'))
len(os.listdir(validation_path+'pedestrian'))

len(os.listdir(validation_path+'no pedestrian'))
# So we have 631 images in our train data and 177 images in the validation data

# Now let us visualaise the average size of the images and try to make all the images in the dimensions 

#for the validation data pedestrians, with a simple for loop we can extract the dimensions

dim1=[]

dim2=[]



for image_filename in os.listdir(validation_path+'pedestrian'):

    

    img=imread(validation_path+'pedestrian/'+image_filename)

    d1,d2,colors=img.shape

    dim1.append(d1)

    dim2.append(d2)
dim1[0:10]
#we have different image sizes, let us visualise more with a joint plot

sns.jointplot(dim1,dim2)
# we have a wide varity of image files are here

np.mean(dim1)
np.mean(dim2)
# let us explore train data also

dim1=[]

dim2=[]



for image_filename in os.listdir(train_path+'pedestrian'):

    

    img=imread(train_path+'pedestrian/'+image_filename)

    d1,d2,colors=img.shape

    dim1.append(d1)

    dim2.append(d2)
sns.jointplot(dim1,dim2)
np.mean(dim1)
np.mean(dim2)
# The image sizes are ranging from 180 to 270, to do a Deep learning network we need to standardize the images

# Here I am going to standardize images to (200,200)

image_shape=(200,200,3)
# we can use image generator to manipulate the images and make it ready for our network

from tensorflow.keras.preprocessing.image import ImageDataGenerator
imread(pedestrian).max()
#maximum value in the cells greater than 1, so we will reshape it in range 0 to 1

# Here we are not doing much since rotation and flippings will not useful for this datatype

image_gen=ImageDataGenerator(rescale=1/255,shear_range=0.1,zoom_range=0.1,fill_mode='nearest')
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(validation_path)
# We will stop preprocessing for now, here we can try so much if we want to improve the accurcy of the model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten

from tensorflow.keras.callbacks import EarlyStopping
# Now let us build our model

model=Sequential()



model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(128,activation='relu')) #same as adding activation in 2nd para

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
early_stop=EarlyStopping(monitor='val_loss',patience=2)
# Batch size is selected to specify how many images are flowing through this network at a time

batch_size=16
train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],color_mode='rgb',

                                                batch_size=batch_size,class_mode='binary')
val_image_gen = image_gen.flow_from_directory(validation_path,target_size=image_shape[:2],color_mode='rgb',

                                               batch_size=batch_size,class_mode='binary',shuffle=False)
train_image_gen.class_indices
results=model.fit_generator(train_image_gen,epochs=15,validation_data=val_image_gen,callbacks=[early_stop])
# Let us visualise the performance

losses=pd.DataFrame(model.history.history)

losses.plot()
# We can see that our accuracy goes up with each epoch and loss come down, and our early callback functioned before overfitting the data
# Our Final Accuracy is

model.evaluate_generator(val_image_gen)
# Let us predict the images

pred=model.predict_generator(val_image_gen)
# our pred will be probability values

pred[:5]
# Let us convert this to predict classes, here I assume if the probability is greater than 0.5 it will be in pedestrian 

# If the probability is less than 0.5 it will be no pedestrian

predictions=pred>0.5
#predictions will be a boolean array

predictions[:5]
#our validation data looks like this

val_image_gen.classes
# Let us evaluate our model

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(val_image_gen.classes,predictions))
confusion_matrix(val_image_gen.classes,predictions)