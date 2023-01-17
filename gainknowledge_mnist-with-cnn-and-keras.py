# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import itertools



import sklearn

import matplotlib as mpl

import matplotlib.pyplot as plt



#%config InlineBackend.figure_formats = {'pdf',}

%matplotlib inline
print(os.listdir("../input/digit-recognizer"))

%time dfLabel = pd.read_csv('../input/digit-recognizer/train.csv')
dfLabel.describe()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

ax.hist(dfLabel['label'],bins=[0,1,2,3,4,5,6,7,8,9,10], edgecolor="b", histtype="bar",align='left')

ax.set_title('Histogram: Training data set')

ax.set(xlabel='Number', ylabel='Frequency')

ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9] );

ax.axhline(y=(dfLabel['label'].size/10), label="average frequency",linestyle='dashed',   color='r')

ax.legend()
%time dfPredict = pd.read_csv('../input/digit-recognizer/test.csv')
dfPredict.head()
dfPredict.describe()
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



dfTmp = dfLabel.copy(deep=True) # # make a copy

tmpLabel = dfTmp['label'] # get the label

label = to_categorical(tmpLabel, num_classes = 10)

del dfTmp['label'] # remove the label column

dfTmp = dfTmp/255 # rescale the data to be between 0 and 1

labeledImage = dfTmp.values.reshape(-1,28,28,1) # convert the data into a tensor



# check the tensor size

assert labeledImage.shape == (dfTmp.shape[0],28,28,1), "The tensor shape {} is not equal to expected tensor size {}".format(labeledImage.shape ,(dfTmp.shape[0],28,28,1))

assert len(label) == dfTmp.shape[0], "The size of the labels {} is not equal to the labeld image size {}".format(len(label),dfTmp.shape[0]) 
def displayData(X,Y):

    # set up array

    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15,15))

    fig.suptitle( "Display randomly images of the training data set")

    # loop over randomly drawn numbers

    for i in range(10):

        for j in range(10):

            ind = np.random.randint(0,high=X.shape[0])

            ax[i,j].set_title("Label: {}".format(np.argmax( Y[ind])))

            ax[i,j].imshow(X[ind][:,:,0], cmap='gray_r') # display it as gray colors.

            plt.setp(ax[i,j].get_xticklabels(), visible=False)

            plt.setp(ax[i,j].get_yticklabels(), visible=False)

    

    fig.subplots_adjust(hspace=0.5, wspace=0.5)



displayData(labeledImage,label) 
dfTmp = dfPredict.copy(deep=True) # make a copy

dfTmp= dfTmp/255 # rescale the data to be between 0 and 1

testImage = dfTmp.values.reshape(-1,28,28,1) # convert the data into a tensor



# check the tensor size

assert testImage.shape == (dfTmp.shape[0],28,28,1), "The tensor shape {} is not equal to expected tensor size {}".format(testImage.shape ,(dfTmp.shape[0],28,28,1))
from sklearn.model_selection import train_test_split

random_state=42 # set a fixed value for reproduceability

# we have the numbers 0 till 9 so we have to ensure that the numbers are evenly distributed.

X_train, X_valid, y_train, y_valid = train_test_split(labeledImage, label, test_size = 0.1, random_state = random_state,  stratify = label)
from keras import layers

from keras import models



model = models.Sequential()



model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(256, activation = "relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation = "softmax"))
# Define the optimizer

import keras 

optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 512
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

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                               epochs = epochs, validation_data = (X_valid,y_valid),

                               verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                               , callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1, figsize=(15,10))

fig.suptitle( "Training & validation curves")

ax[0].set_title("Loss")

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].set_title("Accuracy")

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):

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

Y_pred = model.predict(X_valid)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_valid,axis = 1) 

# compute the confusion matrix

confusion_mtx = sklearn.metrics.confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
test_loss, test_acc = model.evaluate(X_valid, y_valid)

print("The test accuraccy is {}".format(test_acc))
# predict results based on the model

results = model.predict(testImage)

# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission_result = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission_result.to_csv("result.csv",index=False)