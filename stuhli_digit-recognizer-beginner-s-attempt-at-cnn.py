# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting graphs

import matplotlib.image as mpimg # plotting images

%matplotlib inline

import seaborn as sns # more graphs



# some machine learning tools

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# neural network tools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator # for data augmentation

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler # for adapting learning rate



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# adapting plot style

sns.set(style='white', context='notebook', palette='deep')
# Load data

X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')
# training dataset

print(X_train.shape)

print(X_train.info())

print(X_train.head())
# test dataset

print(X_test.shape)

print(X_test.info())

print(X_test.head())
# drop label column and store it as expected output

y_train = X_train.pop('label')



# double check

print(y_train.shape)

print(X_train.shape, X_test.shape)

print(X_train.head())
y_train.value_counts()
# NaN in training input

print(X_train.isnull().values.any())

# NaN in test input

print(X_test.isnull().values.any())

# NaN in training expected output

print(y_train.isnull().values.any())
print(X_train.apply(pd.value_counts))
X_train = X_train / 255.0

X_test = X_test / 255.0



# check values

# print(X_train.apply(pd.value_counts))
# the shape should be 28x28x1 as keras requires an additional dimension for the canal

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



print(X_train.shape)

print(X_test.shape)
y_train = to_categorical(y_train, num_classes=10)
## For now we will use a small training set (and consequently a large validation set)

## to speed up the running time during prototyping

## This will need to be changed before tuning the hyperparameters

## 

## temporary split

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

## replace with final split



# final train-test split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
# create model

model = Sequential()



# (Conv2D -> BatchNormalization) * 2 -> MaxPool2D -> Dropout

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', 

                 input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(4,4), strides=(2,2)))

model.add(Dropout(0.25))



# repeat above sequence

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(4,4), strides=(2,2)))

model.add(Dropout(0.25))



# Flatten -> Dense -> Dropout -> Dense

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(units=10, activation='softmax'))
# the default parameter settings of RMSprop should work fine

# but maybe the learning rate needs to be changed later

optimizer = RMSprop()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,

                              patience=2, min_lr=0.000001, verbose=1)

# some additional parameter settings for the model

epochs = 30

batch_size = 86

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, 

                             height_shift_range=0.1, zoom_range=0.1, 

                             fill_mode='nearest')



datagen.fit(X_train)
fit_model = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                                epochs=epochs, validation_data=(X_val, y_val), verbose=2,

                                steps_per_epoch=X_train.shape[0] // batch_size, 

                                callbacks=[reduce_lr])
# plot the loss functions

plt.plot(fit_model.history['loss'], color='b', label='Training loss')

plt.plot(fit_model.history['val_loss'], color='r', label='Validation loss')

plt.title('Loss functions')

plt.legend()

plt.show()



# plot the development of the model's accuracy

plt.plot(fit_model.history['acc'], color='b', label='Training accuracy')

plt.plot(fit_model.history['val_acc'], color='r', label='Validation accuracy')

plt.title('Accuracy')

plt.legend()

plt.show()

def create_model_confusion_matrix(model, X_input, y_expected):

    """

    This function creates the confusion matrix and plots it.

    """

    # let the model predict the output given X_input

    y_predicted = model.predict(X_input)

    # convert predicted and expected output from one-hot vector to label

    y_predicted_classes = np.argmax(y_predicted, axis=1)

    y_expected_classes = np.argmax(y_expected, axis=1)

    

    # calculate the confusion matrix and convert it

    # to a DataFrame object for plotting

    cm = confusion_matrix(y_expected_classes, y_predicted_classes)

    df_cm = pd.DataFrame(cm, range(10), range(10))

    

    # plot the confusion matrix

    ax = sns.heatmap(df_cm)

    ax.set(xlabel='expected', ylabel='predicted')

    ax.set_title('Confusion Matrix')

    plt.show()

    

    return df_cm



create_model_confusion_matrix(model, X_val, y_val)
# the necessary functions for plotting images along with their 

# predicted and expected labels



def plot_labeled_images(model, X_input, y_expected, mode='random'):

    """

    This function plots a total of 9 images from the given set 

    along with their predicted and expected labels.

    

    The function has two modes: 'random' and 'errors'.

    If mode is set to 'random', random images are plotted.

    If mode is set to 'errors', only images are plotted where the predicted 

    label does not match the expected one.

    """

    num = 9

    if mode == 'random':

        selected_digits = get_random_digits(model, X_input, y_expected, num)

    elif mode == 'errors':

        selected_digits = get_error_digits(model, X_input, y_expected, num)

    else:

        raise ValueError("Unknown value for mode. Only 'random' and 'errors' are accepted.")

    

    # plot the digits

    n = 0

    rows = 3

    cols = 3

    fig, ax = plt.subplots(rows, cols ,sharex=True, sharey=True)

    plt.subplots_adjust(top=1.5) 

    for row in range(rows):

        for col in range(cols):

            ax[row, col].imshow(selected_digits[n][0].reshape((28, 28)))

            ax[row, col].set_title("Predicted label: {}\nExpected label: {}".format(

                selected_digits[n][1], selected_digits[n][2]

            ))

            n +=1



def get_random_digits(model, X_input, y_expected, num):

    """

    This function returns a total of num random digits from the dataset.

    The output is a len(num) tuple of tuples containing an input array, 

    predicted label and expected label each.

    """

    # let the model predict the output given X_input

    y_predicted = model.predict(X_input)

    # convert predicted and expected output from one-hot vector to label

    y_predicted_classes = np.argmax(y_predicted, axis=1)

    y_expected_classes = np.argmax(y_expected, axis=1)

    

    # get num random digits (image, predicted label, expected label)

    digit_sets = get_digit_sets(

        num, X_input, y_expected_classes, y_predicted_classes

    )

    

    return digit_sets



def get_error_digits(model, X_input, y_expected, num):

    """

    This function returns a total of num random digits from the dataset

    where the predicted label does not match the expected one.

    The output is a len(num) tuple of tuples containing an input array, 

    predicted label and expected label each.

    """

    # let the model predict the output given X_input

    y_predicted = model.predict(X_input)

    # convert predicted and expected output from one-hot vector to label

    y_predicted_classes = np.argmax(y_predicted, axis=1)

    y_expected_classes = np.argmax(y_expected, axis=1)

    

    # pick only instances where predicted and expected labels don't match

    errors = (y_predicted_classes - y_expected_classes != 0)

    y_predicted_classes_errors = y_predicted_classes[errors]

    y_expected_classes_errors = y_expected_classes[errors]

    X_input_errors = X_input[errors]

    

    # get num random digits (image, predicted label, expected label)

    digit_sets = get_digit_sets(

        num, X_input_errors, 

        y_expected_classes_errors, y_predicted_classes_errors

    )

    

    return digit_sets

    

def get_digit_sets(num, X_possible, y_expected_classes, y_predicted_classes):

    """

    This function returns a tuple of len(num) containing random digit images

    along with their expected and predicted labels. 

    

    Each entry of the tuple is itself a tuple of the form 

    (image, y_predicted, y_expected).

    """

    indices = np.random.randint(X_possible.shape[0], size=num)

    digit_sets = tuple((X_possible[i], y_predicted_classes[i], y_expected_classes[i])

                      for i in indices)

    return digit_sets

# plot some random images to see whether they are labeled correctly

plot_labeled_images(model, X_val, y_val, mode='random')
plot_labeled_images(model, X_val, y_val, mode='errors')
# predict results

y_test_pred = model.predict(X_test)



# select the indices with the highest probability

# these are our predicted labels

y_test_pred = np.argmax(y_test_pred, axis=1)



# convert to DataFrame object

y_test_pred = pd.Series(y_test_pred, name="Label")



# convert to CSV file as required

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), y_test_pred], axis=1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)