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
file_path = "/kaggle/input/digit-recognizer/train.csv"
X_train = pd.read_csv(file_path)
y_train = X_train.label
X_train = X_train.drop(columns = ["label"])

file_path = "/kaggle/input/digit-recognizer/test.csv"
X_test = pd.read_csv(file_path)
X_test = np.array(X_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
#Normalisation
X_train = X_train /255
X_test = X_test / 255
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
print(X_train.shape)
print(X_test.shape)
from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)

import matplotlib.pyplot as plt
#Test print a digit
g = plt.imshow(X_train[11][:,:,0])
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras.layers.normalization import BatchNormalization
import keras
import tensorflow as tf
import efficientnet.keras as efn
import torchvision.models as models
alexnet = models.alexnet(num_classes = 10)
print(alexnet)

from keras import optimizers
import torch.optim

# Define the optimizer
optimizer = torch.optim.RMSprop(eps=1e-08,#Small value to avoid zero denominator.
                                     weight_decay=0.01)
                                    
# Compile the model
alexnet.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
import torch.optim

#Set a learning rate annealer
#learning_rate_reduction = StepLR(monitor='val_acc', 
                                            #patience=3, #number of epochs with no improvement after which learning rate will be reduced.
                                            #verbose=1, #Display a message after each epoch
                                            #factor=0.5, 
                                            #min_lr=0.00001)

learning_rate_reduction = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2)
epochs = 40
batch_size = 50
# With data augmentation to prevent overfitting (accuracy 0.99286)
datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
# Fit the model
history = alexnet.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
# predict results
results = alexnet.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

#Save the predictions to a csv

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

#submission = pd.DataFrame({'ImageId': X_test.index + 1, 'Label': results})
submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")