# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from tensorflow import keras



from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import RMSprop

from keras import regularizers

#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#get data sets

train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_df.head()
img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255.0

    return out_x, out_y



X_train, Y_train = prep_data(train_df.values)

#show image

print(Y_train[0])

plt.imshow(X_train[0][:,:,0])
#creating CNN network

img_rows , img_col = 28,28

dig_model = Sequential()



#add first layer

dig_model.add(Conv2D(32,kernel_size=(3,3),

                    activation = 'relu',input_shape = (img_rows,img_col,1)))

######hidden layers######



dig_model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))

dig_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

dig_model.add(MaxPool2D(pool_size = (2,2)))

dig_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

dig_model.add(MaxPool2D(pool_size = (2,2)))

dig_model.add(Flatten())

dig_model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.001)))

dig_model.add(Dropout(0.5))

dig_model.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.001)))

dig_model.add(Dense(num_classes,activation='softmax'))



######compile the model#####

dig_model.compile(loss="categorical_crossentropy",

                     optimizer = RMSprop(lr=0.001),

                     metrics = ['accuracy'])



# fit the model and see accuacy and loss values

hist = dig_model.fit(X_train,Y_train,batch_size=100,epochs=30,validation_split=0.2)
def hist_acc(history):

    # summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

def hist_loss(history):

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

    

#plot graphs of accuracy and loss

hist_acc(hist)

hist_loss(hist)
#predict training data ny the model

Y_pred  = dig_model.predict(X_train)
#predict training data

Y_pred_val = np.argmax(Y_pred,axis =1)

Y_train_val = np.argmax(Y_train,axis =1)
errors = Y_pred_val-Y_train_val !=0

Y_pred_val[errors]

plt.imshow(X_train[3200][:,:,0])
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_val - Y_train_val != 0)



Y_pred_classes_errors = Y_pred_val[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_train_val[errors]

X_val_errors = X_train[errors]



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

            ax[row,col].set_title("\n Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

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

#set test data for prediction

x = test_df.values

num_images = test_df.values.shape[0]

X_test = x.reshape(num_images, img_rows, img_cols, 1)

#normelize values

X_test = X_test / 255.0
# predict results

results = dig_model.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)