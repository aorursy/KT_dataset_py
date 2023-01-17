import os

import numpy as np # linear algebra, matrix multiplications

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

xtrain = []

for imgfolder in os.listdir('../input/data/train/'): #iterate thru each of the 5 celeb folders

    for filename in os.listdir('../input/data/train/' + imgfolder):# iterate thru each image in a celeb folder

        filename = '../input/data/train/' + imgfolder + '/' + filename # build the path to the image file

        img = cv2.imread(filename,0) # read the image using OpenCV

        img = cv2.resize(img, (47,62), interpolation = cv2.INTER_AREA)

        xtrain.append(img)

xtrain = np.asarray(xtrain)

xtrain.shape
# building an 1D array of labels for the training dataset

ytrain = []

for imgfolder in os.listdir('../input/data/train/'):

    for filename in os.listdir('../input/data/train/' + imgfolder):

        ytrain.append(imgfolder)

ytrain = np.asarray(ytrain)

ytrain.shape
import cv2

xtest = []

for imgfolder in os.listdir('../input/data/val/'): #iterate thru each of the 5 celeb folders

    for filename in os.listdir('../input/data/val/' + imgfolder):# iterate thru each image in a celeb folder

        filename = '../input/data/val/' + imgfolder + '/' + filename # build the path to the image file

        img = cv2.imread(filename,0) # read the image using OpenCV

        img = cv2.resize(img, (47,62), interpolation = cv2.INTER_AREA)

        xtest.append(img)

xtest = np.asarray(xtest)

xtest.shape
#Building a 1D array of test labels

ytest = []

for imgfolder in os.listdir('../input/data/val/'):

    for filename in os.listdir('../input/data/val/' + imgfolder):

        ytest.append(imgfolder)

ytest = np.asarray(ytest)
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')



# plot how many images there are in each class

sns.countplot(ytrain)



print(ytrain.shape)

print(type(ytrain))



# array with each class and its number of images

vals_class = ytrain.value_counts()

print(vals_class)



# mean and std

cls_mean = np.mean(vals_class)

cls_std = np.std(vals_class,ddof=1)



print("The mean amount of elements per class is", cls_mean)

print("The standard deviation in the element per class distribution is", cls_std)



# 68% - 95% - 99% rule, the 68% of the data should be cls_std away from the mean and so on

# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule

if cls_std > cls_mean * (0.6827 / 2):

    print("The standard deviation is high")

    

# if the data is skewed then we won't be able to use accurace as its results will be misleading and we may use F-beta score instead.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



# convert train dataset to (num_images, img_rows, img_cols) format in order to plot it

xtrain_vis = xtrain.values.reshape(ntrain, dim, dim)



# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html

# subplot(2,3,3) = subplot(233)

# a grid of 3x3 is created, then plots are inserted in some of these slots

for i in range(0,9): # how many imgs will show from the 3x3 grid

    plt.subplot(330 + (i+1)) # open next subplot

    plt.imshow(xtrain_vis[i], cmap=plt.get_cmap('gray'))

    plt.title(ytrain[i]);
# Normalize the data

xtrain = xtrain / 255.0

xtest = xtest / 255.0
# reshape of image data to (nimg, img_rows, img_cols, 1)

def df_reshape(df):

    print("Previous shape, pixels are in 1D vector:", df.shape)

    df = df.values.reshape(-1, dim, dim, 1) 

    # -1 means the dimension doesn't change, so 42000 in the case of xtrain and 28000 in the case of test

    print("After reshape, pixels are a 28x28x1 3D matrix:", df.shape)

    return df



xtrain = df_reshape(xtrain) # numpy.ndarray type

xtest = df_reshape(xtest) # numpy.ndarray type
from keras.utils.np_utils import to_categorical



print(type(ytrain))

# number of classes, in this case 10

nclasses = ytrain.max() - ytrain.min() + 1



print("Shape of ytrain before: ", ytrain.shape) # (42000,)



ytrain = to_categorical(ytrain, num_classes = nclasses)



print("Shape of ytrain after: ", ytrain.shape) # (42000, 10), also numpy.ndarray type

print(type(ytrain))
from sklearn.model_selection import train_test_split



# fix random seed for reproducibility

seed = 2

np.random.seed(seed)



# percentage of xtrain which will be xval

split_pct = 0.2



# Split the train and the validation set

xtrain, xval, ytrain, yval = train_test_split(xtrain,

                                              ytrain, 

                                              test_size=split_pct,

                                              random_state=seed,

                                              shuffle=True,

                                              stratify=ytrain

                                             )



print("Training", xtrain.shape, ytrain.shape, "\t Testing", xval.shape, yval.shape)
from keras import backend as K



# for the architecture

from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPool2D, AvgPool2D



# optimizer, data generator and learning rate reductor

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()



dim = 28

nclasses = 10



model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(dim,dim,1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu',))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(84, activation='relu'))

model.add(Dense(nclasses, activation='softmax'))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
lr_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                 patience=3, 

                                 verbose=1, 

                                 factor=0.5, 

                                 min_lr=0.00001)
datagen = ImageDataGenerator(

          featurewise_center=False,            # set input mean to 0 over the dataset

          samplewise_center=False,             # set each sample mean to 0

          featurewise_std_normalization=False, # divide inputs by std of the dataset

          samplewise_std_normalization=False,  # divide each input by its std

          zca_whitening=False,                 # apply ZCA whitening

          rotation_range=30,                   # randomly rotate images in the range (degrees, 0 to 180)

          zoom_range = 0.1,                    # Randomly zoom image 

          width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)

          height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)

          horizontal_flip=False,               # randomly flip images

          vertical_flip=False)                 # randomly flip images



#datagen.fit(xtrain)
epochs = 15

batch_size = 1024
# history = model.fit_generator(datagen.flow(xtrain,ytrain, batch_size=batch_size),

#                               epochs=epochs, 

#                               validation_data=(xval,yval),

#                               verbose=1, 

#                               steps_per_epoch=xtrain.shape[0] // batch_size, 

#                               callbacks=[lr_reduction])





history = model.fit(x=xtrain, 

                    y=ytrain, 

                    batch_size=batch_size, 

                    epochs=epochs, 

                    verbose=1,

                    callbacks=[lr_reduction],

                    validation_data=(xval,yval), 

                    shuffle=True)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

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

ypred_onehot = model.predict(xval)

# Convert predictions classes from one hot vectors to labels: [0 0 1 0 0 ...] --> 2

ypred = np.argmax(ypred_onehot,axis=1)

# Convert validation observations from one hot vectors to labels

ytrue = np.argmax(yval,axis=1)

# compute the confusion matrix

confusion_mtx = confusion_matrix(ytrue, ypred)

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes=range(nclasses))
errors = (ypred - ytrue != 0) # array of bools with true when there is an error or false when the image is cor



ypred_er = ypred_onehot[errors]

ypred_classes_er = ypred[errors]

ytrue_er = ytrue[errors]

xval_er = xval[errors]



def display_errors(errors_index, img_errors, pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1

            

# Probabilities of the wrong predicted numbers

ypred_er_prob = np.max(ypred_er,axis=1)



# Predicted probabilities of the true values in the error set

true_prob_er = np.diagonal(np.take(ypred_er, ytrue_er, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_er = ypred_er_prob - true_prob_er



# Sorted list of the delta prob errors

sorted_delta_er = np.argsort(delta_pred_true_er)



# Top 6 errors. You can change the range to see other images

most_important_er = sorted_delta_er[-6:]



# Show the top 6 errors

display_errors(most_important_er, xval_er, ypred_classes_er, ytrue_er)
from sklearn.metrics import accuracy_score 

#xtest = xtest.reshape(-1, dim, dim, nchannels)

ypredtest = model.predict_classes(xtest)

print("test_acc", accuracy_score(ytest.values, ypredtest))