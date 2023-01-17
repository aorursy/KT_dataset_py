 !unzip autonobot-detection-challenge-0.zip    # unzip the file 
# Setting the seed

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)
import numpy as np

import pandas as pd

from imageio import imread

from skimage.transform import resize

import os

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras



train = pd.read_csv('training.csv')

newid = [str(i) for i in train['Id']]



width, height = 512,512



file = os.listdir()



# Loading the training Images

images = [imread('Images/Images/Training/' + j) for j in newid]

resized = [resize(i, (width, height)) for i in images]

images = np.array(resized)

# Loading the testing images

testid=pd.read_csv('sample.csv',usecols=['Id'])

testid = [str(i) for i in testid['Id']]

width, height = 512,512

images_test = [imread('Images/Images/Testing/' + j) for j in testid]

resized_test = [resize(i, (width, height)) for i in images_test]

images_test = np.array(resized_test)
# Plotting a sample of training images with label

train_0 = train[train['Category'] == 0]

train_1 = train[train['Category'] == 1]



train_labels = [0,1]

class_names=['no cone','cone']



plt.figure(figsize=(14,14))

for i in range(100):

    plt.subplot(10,10,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train['Category'].values[i]])

plt.show()
from keras import regularizers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

#from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = images[0].shape



classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(16, (3, 3),input_shape = input_shape,activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3),input_shape = input_shape, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a third convolutional layer

classifier.add(Conv2D(64, (3, 3), input_shape = input_shape,activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu',kernel_regularizer=regularizers.l2(0.05)))    # adding a penalty to avoid over-fitting by using l2 reguolarizer

classifier.add(Dense(units = 1, activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.01)))   # sigmoid as our output is either 0 or 1



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(images,train.Category.values, epochs=120, batch_size=32,validation_split=0.1)   # taking 10% of training data as validation set

# Predicting the label for test images 

result = classifier.predict(images_test)
# Converting the label dtype and using round function to get label as 0 or 1

result=np.round(result).astype('int64')
df_sub=pd.DataFrame(testid,columns=['Id'])

df_sub['Category']=result
df_sub.info()
df_sub.to_csv('submission.csv',index=False)    # Exporting the submission file 
# saving the model

classifier.save('autonobot_trial_model_er_8.h5')  # can reload the model and run to replicate the results
classifier.summary()
# Reloading the model 

import tensorflow as tf 

new_model=tf.keras.models.load_model('autonobot_trial_model_er_8.h5')

new_model.optimizer   # same as our aforementioned model's optimizer
#result2 = new_model.predict(images_test)    # code to replicate the results