# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Flatten, Activation, MaxPool2D
#Image augmentation and preprocessing

#rescaling of pixel values for efficiency , data augmentation which is desiarable like zoom and horizontal flip
train_pr = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, zoom_range=0.2,horizontal_flip=True)

#rescaled test data, test data should not be augmented 
test_pr = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#collection from subdirectories with folder name as labels and images as features 
training_data = train_pr.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',
	                                              batch_size=32,target_size=(150, 150),class_mode='categorical')  #categorical: one_hot_encoding
#(150,150) is with reference to data info given with dataset.
test_data = test_pr.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',
                                                         batch_size=32,target_size=(150, 150),class_mode='categorical')   #categorical: one_hot_encoding
#checking shape of the data to cross verify and init useful var
batch_size, row, height, channel = training_data[0][0].shape


classes = ["building", "forest", "glacier", "mountain", "sea", "street"]  #for easy labelling


print("number of batches ", len(training_data))
print("shape of each batch: ",(batch_size, row, height, channel))
print("shape of labels for each batch: ", training_data[0][1].shape)

#inference
# training data[0] means first batch, training data[0][1] means labels(hot_encoding) for 1st batch and training data[0][0] means features for first batch
#plotting data, for checking any abnormality and how to proceed.

#number of images per row and number of rows
n = 3      # n should be less than 32
#syntax for create subplots
fig, axs = plt.subplots(n,n)
fig.tight_layout()
#next image after each iter
c = 1
for i in range(n):
    for j in range(n):
        #selected a batch
        axs[i,j].imshow(np.reshape(training_data[0][0][c], [150,150,3]))
        axs[i,j].set_title(classes[np.argmax(training_data[0][1][c])])
        c += 1
#exponentialy depth: increase of number of filter and exponential dimensionality reduction of image.... approach  with repition = 1
#dropout for regularization since model is likely to overfit
#batch normalization for efficiency and to avoid exploding or vanishing values.

model = tf.keras.Sequential([
    Conv2D(16, kernel_size=3, activation='relu', input_shape=(150,150,3), padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(32, kernel_size=3, activation='relu', padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=3, activation='relu', padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(128, kernel_size=3, activation='relu', padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(256, kernel_size=3, activation='relu', padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(512, kernel_size=3, activation='relu', padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(1024, kernel_size=3, activation='relu', padding = "SAME"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation = "relu"),
    Dropout(0.5),
    Dense(6, activation = "softmax")
])

#added a dropout layer to reduce overfitting.

#check the model.. 6.4M params.
model.summary()
#reduces learning rate if no significant decrease in loss for {{patience}} epochs.
RLROP = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.1,min_lr=1e-8)

#save best model
#path where highest valid accuracy will save
checkpoint_file = "model.hdf5"
#callback to save the best model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_file,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max', verbose = 1,
    save_best_only=True)



#adam, binary_crossentropy, accuracy
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

#training the model, using RLROP and model_checkpoint_callback as callback
#verbose = 1 to take the loss data for each epoch for plotting and experimenting
history = model.fit_generator(
    training_data, 
    validation_data  = test_data,
    epochs = 30, verbose = 1,
    callbacks = [RLROP, model_checkpoint_callback]
)
#the initial drop in accuracy is due to changing learning rate and dropout fluctuations.
#achieved local minima without overfitting.
#loading the best model
model.load_weights(checkpoint_file)       #26th epoch model
#checking accuracy on valid set
results = model.evaluate_generator(test_data)
#printing the accuracy
print(round(results[1],3)*100,"%: accuracy on test data")   #achieved in 26th epoch
#plotting the loss throughout the epochs
#plotting train loss graph
plt.plot(history.history['loss'], label='train loss')
#plotting test loss
plt.plot(history.history['val_loss'], label='test loss')
#setting title
plt.title('classification')
#setting y axis label
plt.ylabel('loss')
#setting x axis label
plt.xlabel('No. of epochs')
#setting legend: explaining which color represents which graph
plt.legend(loc="upper left")
plt.show()

#ams grad and ReduceLROnPlateau creates some fluctuation but epochs seems stable at the end.
import random
#select random batch each time
ran = random.randint(0,92)
#find predictions for the batch
result = model.predict(test_data[ran][0])
c = 0

#check the images which are predicted wrong
for i in range(32):
    #if predicted is different than actual
    if np.argmax(test_data[ran][1][i]) != np.argmax(result[i]):
        #show the image
        plt.imshow(test_data[ran][0][i])
        #set title referring both actual and predicted class
        plt.title("true: "+ classes[np.argmax(test_data[ran][1][i])] + "::: predicted: " + classes[np.argmax(result[i])])
        c += 1
#print total error in batch selected
print("faults " , c)
#the errors seems reasonable... isn't it? 
"""-:Sambhav Gupta, DPS RK Puram"""
""" samg2003 """
"""generalised CNN model and procedure for Supervised learning on small data and little computation"""