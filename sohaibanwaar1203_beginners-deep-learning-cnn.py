# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()

test.head()


# taking 2nd Row of Dataframe

img=train.iloc[1][1:785].values

label=train.iloc[1][0]

# we are Excluding Coloum 0 because it indicates labels we are just taking coloums from 1 to 785

print("Shape of Image ",img.shape)

# Images are aleays in 3 Diminsioon

# 1st diminsion represents width of the image

# 2nd diminsion represents height of the image

# 3rd represents is it a colored image or Grayscale Image

# IF the value of 3rd diminsion is 1 it means it is Gray Scale 

# If the value if 3rd diminsion is 3 it means it is Colored because it indicates 3 values

# i.e (red,blue,green) Colours are always the combination of Red blue green. 



# For example Image diminsion is (32,32,3) etc like that means 32 height 32 width and 3 

#means colorful Image



# So now to Visualize Image we have to convert our image to 3 diminsions. basic Diminsion of 

# Mnist image is 28*28*1=784 means 28 height 28 width and 1 means grayScale Images



img=img.reshape(28,28)

# if we didnt write 1 thats not a problem in grayscale image but if you are working on color images

# must write 3

print("28*28 is 784")

print("New Diminsion of Image" ,img.shape)



import matplotlib.pyplot as plt



fig = plt.figure()

plt.imshow(img,cmap='gray')

fig.suptitle("Label of this Image is: "+ str(label))



plt.show()
for i in range(0,10) :   

    img=train.iloc[i][1:785].values

    label=train.iloc[i][0]

    img=img.reshape(28,28)

    fig = plt.figure()

    plt.imshow(img,cmap='gray')

    fig.suptitle("Label of this Image is: "+ str(label))

    

    plt.show()
# import the necessary packages

from keras.models import Sequential



from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers import Dropout

from keras.layers.core import Flatten

from keras.layers.core import Dense

from sklearn.model_selection import train_test_split

from sklearn import datasets

from keras.optimizers import adam

from keras.utils import np_utils

import numpy as np

import cv2

 

#copying all the values to X except coloum 1 because those are our labels which we will

# copy in y

X = (train.iloc[:,1:].values)

#We separate the target variable

y = train.iloc[:,0].values

#We get the test data

test = test
#We reshape the train set

X = X.reshape((X.shape[0], 28, 28))

X = X[:, :, :, np.newaxis]

y=np_utils.to_categorical(y, 10)

#We reshape the test set

test=np.asarray(test)

test = test.reshape((test.shape[0], 28, 28))

#test = test[:, :, :, np.newaxis]



print("Shape of X",X.shape)

print ("Shape of y",y.shape)

print("Shape of Test",test.shape)
#We split the train set and test set so that we can evaluate our model

X_train, X_test, y_train, y_test=train_test_split(

    X , y, test_size=0.33)

X_train =X_train/ 255

X_test = X_test/255
# initialize the model

height=28

width=28

depth=1

classes=10





model = Sequential()

# first set of CONV => RELU => POOL

model.add(Convolution2D(20, 2, 2, border_mode="same",input_shape=(height, width,depth)))

model.add(Activation("relu"))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL

model.add(Convolution2D(50, 2, 2, border_mode="same"))

model.add(Activation("relu"))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Third set of CONV => RELU => POOL

model.add(Convolution2D(100, 2, 2, border_mode="same"))

model.add(Activation("relu"))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers

model.add(Flatten())

model.add(Dense(500))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(250))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(100))

model.add(Activation("relu"))

model.add(Dense(50))

model.add(Dropout(0.3))

model.add(Activation("relu"))

model.add(Dense(32))

model.add(Dropout(0.2))

model.add(Activation("relu"))

model.add(Dense(classes))

model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam",

metrics=["accuracy"])

        

        









 

# show the accuracy on the testing set

history=model.fit(X_train,y_train, epochs = 70 ,batch_size = 128,validation_data=(X_test,y_test))











# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])





plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
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

Y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
X=X/225

model.fit(X,y, epochs = 100 ,batch_size = 128,verbose=0)
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_test[errors]



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
# predict results

test=test.reshape(-1,28,28,1)

test=test/225

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)