# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape, test.shape
# ASSIGN X_train AND y_train

X_train = train.drop('label',axis = 1)

y_train = train.label
y_train.value_counts()
import seaborn as sns 

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 15, 9



sns.countplot(y_train);
# Checking for null values in train and test

X_train.isnull().any().sum()
test.isnull().any().sum()
X_train /= 255.

test /= 255.
# Setting height and widht to 28px and canal to 1



X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
# One Hot Encoding target values

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)
# Create train and validation sets

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state= 3)
# Example

g = plt.imshow(X_train[0][:,:,0])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
# Assigning the Optimizer

from keras.optimizers import Adamax

optimizer = Adamax(lr=0.001)
# Compiling the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

                             rotation_range=10,

                             zoom_range = 0.1,

                             width_shift_range=0.1,

                             height_shift_range=0.1)



datagen.fit(X_train)
%%time



model.fit_generator(datagen.flow(X_train,y_train, batch_size= 86),

                                  epochs = 100, validation_data = (X_val,y_val),

                                  verbose = 2, steps_per_epoch=X_train.shape[0] // 86,

                                  callbacks=[learning_rate_reduction])
from sklearn.metrics import confusion_matrix



# Predict the validation dataset

y_pred = model.predict(X_val)

# Convert predictions to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

sns.heatmap(confusion_mtx, annot=True, fmt='d');
# Difference between predicted labels and true labels

errors = (y_pred_classes - y_true != 0)



y_pred_classes_errors = y_pred_classes[errors]

y_pred_errors = y_pred[errors]

y_true_errors = y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

y_pred_errors_prob = np.max(y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)
# predict results

results = model.predict(test)



# change predictions to one hot vectors

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)