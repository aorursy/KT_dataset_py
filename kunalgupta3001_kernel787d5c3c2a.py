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
import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import models, layers, optimizers

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, KFold

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)

from tensorflow.keras import regularizers
train_orig=pd.read_csv('../input/digit-recognizer/train.csv')

test_orig=pd.read_csv('../input/digit-recognizer/test.csv')

sam_orig=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train_x , test_x =  train_test_split(train_orig , test_size=0.1 , random_state=1)

train_y , test_y = train_x.pop('label') , test_x.pop('label')

train_x , test_x = train_x.values , test_x.values

test_orig=test_orig.values

train_x.shape
train_x=train_x.reshape(37800 , 28 , 28)

for i in range(9):

    plt.imshow(train_x[i])

    plt.show()

    
train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))

test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

test_orig = test_orig.reshape((test_orig.shape[0], 28 , 28 , 1))

train_x, test_x, test_orig = train_x / 255.0, test_x / 255.0, test_orig/ 255.0

                              





train_y = to_categorical(train_y)

test_y = to_categorical(test_y)

train_x.shape, test_x.shape, train_y.shape, test_y.shape

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",

                           kernel_initializer="he_uniform", input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform"),

    tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"),

    tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Flatten(),

     tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99),

    tf.keras.layers.Dense(256, activation="relu",kernel_initializer="he_uniform"),

    tf.keras.layers.Dense(128, activation="relu",kernel_initializer="he_uniform"),

    tf.keras.layers.Dense(84, activation="relu",kernel_initializer="he_uniform"),

    tf.keras.layers.Dense(10, activation="softmax")

])





model.compile(optimizer=tf.keras.optimizers.Adam(

    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,

    name='Adam'

),

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=["accuracy"])



datagen = ImageDataGenerator(

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





datagen.fit(train_x)
reduce_lr = ReduceLROnPlateau(monitor='Test_loss', factor=0.2,

                              patience=3, min_lr=0.00001)

history = model.fit(train_x, train_y,batch_size=256, epochs=50, validation_data=(test_x, test_y), verbose=0)



score1 = model.evaluate(train_x, train_y, verbose=0)

score = model.evaluate(test_x, test_y, verbose=0)

print(f"Train_loss: {score1[0]} / Train accuracy: {score1[1]}")

print(f"Test_loss: {score[0]} / Test accuracy: {score[1]}")
hist = pd.DataFrame(history.history)

hist["epoch"] = history.epoch



def plot_history(history):

    plt.figure()

    plt.xlabel("Epochs")

    plt.ylabel("Train, Val Accuracy")

    plt.plot(hist["epoch"], hist["accuracy"], label="Train acc")

    plt.plot(hist["epoch"], hist["val_accuracy"], label="Val acc")

    plt.legend()

    

    plt.figure()

    plt.xlabel("Epochs")

    plt.ylabel("Train, Val Loss")

    plt.plot(hist["epoch"], hist["loss"], label="Train Loss")

    plt.plot(hist["epoch"], hist["val_loss"], label="Val loss")

    plt.legend()

    plt.show()

    



plot_history(history)
from sklearn.metrics import confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(test_x)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(test_y,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = test_x[errors]



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

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
results = model.predict(test_orig)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)