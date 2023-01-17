import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mpimg

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import MaxPool2D

from keras.layers import Conv2D

from keras.layers import Dense

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
test.shape

train.head()
y_train = train['label']



# Drop the Label column from train dataset so that we get independent variable in X

X_train = train.drop(['label'], axis = 1 )



# free some space

del train



tr = sns.countplot(y_train)

tr
y_train.value_counts()   # Decreasing Order
X_train.isnull().sum().describe()
X_train.isnull().any().describe()    # No missing Value found
test.isnull().any().describe()    # No missing Value found
X_train = X_train/255

test = test/255
X_train = X_train.values.reshape(-1,28,28,1)



#The -1 can be thought of as a flexible value for the library to fill in for you. 

# The restriction here would be that the inner-most shape of the Tensor should be (28, 28, 1). 

# Beyond that, the library can adjust things as needed. In this case, that would be the # of examples in a batch.



test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train,num_classes=10)   #from keras.utils.np_utils import to_categorical
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, random_state = 2)
tr = plt.imshow(X_train[0][ :, : , 0] )
tr = plt.imshow(X_train[98][ :, : , 0] )
obj = Sequential()



obj.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu',input_shape = (28,28,1)))

obj.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))

        

obj.add(MaxPool2D(pool_size=(2,2))) 

obj.add(Dropout(0.25))



obj.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

obj.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

obj.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

obj.add(Dropout(0.25))



obj.add(Flatten())

obj.add(Dense(256, activation = "relu"))

obj.add(Dropout(0.5))

obj.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

obj.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate = ReduceLROnPlateau(monitor='val_acc', patience=3,verbose=1,factor=0.5,min_lr=0.00001)



epochs = 1

batch_size = 86
# With data augmentation to prevent overfitting (accuracy 0.99286)



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
history = obj.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate])
# Without Data Augmentation

hist = obj.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 

         validation_data = (X_val, y_val), verbose = 2)
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

y_pred = obj.predict(X_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
# Display some error results 



# Errors are difference between predicted labels and true labels

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

results = obj.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)