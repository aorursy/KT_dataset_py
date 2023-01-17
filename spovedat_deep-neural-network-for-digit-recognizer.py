import os

import numpy as np

import pandas as pd

from tensorflow.python import keras

from tensorflow.keras import datasets, layers, models

from keras.regularizers import l2

import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load the train and test files



train_img= pd.read_csv('../input/digit-recognizer/train.csv')

test_img= pd.read_csv('../input/digit-recognizer/test.csv')

num_labels= 10
train_img.head(3)
test_img.head(3)
# Function to prepare the train set



def prep(data):

    y= keras.utils.to_categorical(data.label, num_labels)

    

    n_images= data.shape[0]

    x_arr= data.values[:,1:]

    x_arr_sh= x_arr.reshape(n_images, 28, 28, 1)

    x= x_arr_sh/255

    return x,y
# Train data and labels



x, y= prep(train_img) 





# Prepare the test set



n_img= test_img.shape[0]

x_arr_sh= test_img.values[:,:].reshape(n_img, 28, 28, 1)

test_x= x_arr_sh/255
# Preview 5 images and labels from the train set 



fig, ax= plt.subplots(1, 5, constrained_layout= True, figsize= (15, 15))

for i in range(5):

    ax[i].imshow(x[i].reshape(28, 28))

    ax[i].set_xlabel('label:   ' + str(train_img.label[i]), fontsize= 20)
# Create and train the model



model= models.Sequential()



model.add(layers.Conv2D(16, kernel_size= (3,3), kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), input_shape= (28, 28, 1)))



model.add(layers.Conv2D(32, kernel_size= (3,3), strides= 2, kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), activation= 'relu'))

model.add(layers.Conv2D(32, kernel_size= (3,3), strides= 2, kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), activation= 'relu'))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(64, kernel_size= (3,3), kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), activation= 'relu'))

model.add(layers.Conv2D(64, kernel_size= (3,3), kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), activation= 'relu'))

model.add(layers.Dropout(0.25))



model.add(layers.Conv2D(128, kernel_size= (1,1), padding='same', kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), activation= 'relu'))

model.add(layers.Conv2D(128, kernel_size= (1,1), padding='same', kernel_initializer= 'he_normal', kernel_regularizer=l2(0.0005), activation= 'relu'))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(512, kernel_initializer= 'he_normal', activation= 'relu'))

model.add(layers.Dense(num_labels, activation= 'softmax'))



model.compile(loss= keras.losses.categorical_crossentropy, optimizer= 'adam', metrics= ['accuracy'])

hist= model.fit(x, y, batch_size= 64, epochs= 10, steps_per_epoch= (len(x)*0.8) / 64, validation_split = 0.2)

# Visualize the accuracy of the model at each epoch



fig, ax= plt.subplots(1, 2, constrained_layout= True, figsize= (15,5))



ax[0].plot(hist.history['accuracy'])

ax[0].set_title('Train set', fontsize= 15)

ax[0].set_ylabel('accuracy')

ax[0].set_xlabel('epoch')



ax[1].plot(hist.history['val_accuracy'])

ax[1].set_title('Validation set', fontsize= 15)

ax[1].set_ylabel('accuracy')

ax[1].set_xlabel('epoch')
# Make the predictions for the test set



test_y= model.predict(test_x)

pred_y= np.argmax(test_y, axis=1)
# Preview 5 images and labels from the test set 



fig, ax= plt.subplots(1, 5, constrained_layout= True, figsize= (15, 15))

for i in range(5):

    ax[i].imshow(test_x[i].reshape(28,28))

    ax[i].set_xlabel('predicted:   ' + str(pred_y[i]), fontsize= 20)
# Save the predictions as csv



subm= pd.read_csv('../input/digit-recognizer/sample_submission.csv')

subm['Label']= pred_y

subm.to_csv('subm.csv', index= False)