# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from keras.applications.vgg16 import decode_predictions

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

from keras.callbacks import ReduceLROnPlateau



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import matplotlib.pyplot as plt 

from PIL import Image 

import seaborn as sns

import pandas as pd 

import numpy as np 

import os 

import os

import pathlib
def _load_image(img_path):

    img = image.load_img(img_path, target_size=(320, 240))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img
paths = []

types = []

images = []

for dirname, _, filenames in os.walk('/kaggle/input/kdcm32/flowers'):

    for filename in filenames:

        imgpath = os.path.join(dirname, filename)

        suffix = pathlib.Path(imgpath).suffix

        if suffix == '.jpg':

            base_dir = os.path.basename(dirname)

            if base_dir != 'test':

                types.append(base_dir)

                path = os.path.join(dirname, filename)

                paths.append(path)

                loaded_image = _load_image(path)

                images.append(loaded_image)

        else:

            print('not image: ', imgpath)

# Any results you write to the current directory are saved as output.

images = np.vstack(images)
encoder = LabelBinarizer()

types_int = encoder.fit_transform(types)
def _get_model2():

    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu', input_shape = (320,240,3)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

#     model.add(Dropout(0.2))



    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#     model.add(Dropout(0.2))



    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#     model.add(Dropout(0.2))

    

    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#     model.add(Dropout(0.2))

    

    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#     model.add(Dropout(0.2))

    

    model.add(Flatten())

    model.add(Dense(4096, activation = "relu"))

    model.add(Dense(2096, activation = "relu"))

    model.add(Dense(1000, activation = "relu"))

    model.add(Dense(100, activation = "relu"))

    model.add(Dropout(0.2))

    model.add(Dense(5, activation = "softmax"))



    return model

epochs = 30

batch_size = 50

random_seed = 2



model = _get_model2()



# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

optimizer = SGD(lr=0.001, nesterov=True)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



X_train = images

Y_train = types_int

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)



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





datagen.fit(X_train)



# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])



# history = model.fit(x = X_train, y = Y_train, batch_size = batch_size,

#                    epochs = epochs, verbose = 2, validation_data = (X_val, Y_val), callbacks=[learning_rate_reduction])



# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 



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

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = list(set(types))) 
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

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

            ax[row,col].imshow((img_errors[error]).reshape((480,480)))

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
# load test data 

paths_test = []

files_test = []

types_test = []

images_test = []

for dirname, _, filenames in os.walk('/kaggle/input/kdcm32/test'):

    for filename in filenames:

        imgpath = os.path.join(dirname, filename)

        suffix = pathlib.Path(imgpath).suffix

        if suffix == '.jpg':

            types.append(os.path.basename(dirname))



            path = os.path.join(dirname, filename)

            paths_test.append(path)

            image_filename = os.path.basename(path)

            files_test.append(image_filename)

            loaded_image = _load_image(path)

            images_test.append(loaded_image)

        else:

            print('not image: ', imgpath)


def _get_predictions(_model, imgs):

    predictions = []

    for img in imgs:

        pred = _model.predict(img)[0]

        predictions.append(pred)



    return predictions
# predict test data

results = _get_predictions(model, images_test)

# select the index with the maximum probability

indexes = np.argmax(results, axis = 1)

indexes = pd.Series(indexes,name="index")



labels = []

for index in indexes:

    labels.append(encoder.classes_[index])

labels_results = pd.Series(labels ,name="species")



names = pd.Series(files_test, name="name")

submission = pd.concat([names, labels_results], axis = 1)

# print(submission)

submission.to_csv("sub.csv",index=False)