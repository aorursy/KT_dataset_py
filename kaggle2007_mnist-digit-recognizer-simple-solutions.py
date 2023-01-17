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
#################################

# This is a Digit Recognizer (0... 9) Python program

# First couple of version submitted with a simple Neural Network

# Later Convoluted Neural Network (CNN) was introduced

#################################

import numpy as np

import pandas as pd

from keras.utils import np_utils

import matplotlib.pyplot as plt

from keras import optimizers

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import Dropout, Flatten, Activation, Dense

from keras.layers.normalization import BatchNormalization
##########################

# MNIST data set file : Train Set

##########################



train_mnist_data_X = pd.read_csv('../input/train.csv')

train_mnist_data_X.head()

train_mnist_data_X.info()

#print(train_mnist_data_X.describe()) #shows that these data are quite parsed.

# mean       4.456643

# std        2.887730
###########################

# Taking the label out of the training data

###########################



train_data_Y = train_mnist_data_X["label"]

train_mnist_data_X.drop(["label"], axis = 1, inplace=True)
#############################

# Data Transformation

#############################

train_mnist_data_X.shape # (42000, 784)

train_data_Y.shape # (42000,)



# normalize inputs from 0-255 to 0-1

X_train = train_mnist_data_X / 255



# one hot encode outputs

y_train = np_utils.to_categorical(train_data_Y)



print("--X_train.shape---", X_train.shape) #(42000, 784)

print("--y_train.shape---", y_train.shape) #(42000, 10)
def CNN_Model():

    # create model



    model = Sequential()

    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu', kernel_initializer='normal')) #This expects 4D array input

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3, 3), activation='relu',padding='same',kernel_initializer='normal'))

    model.add(Conv2D(32, (3, 3), activation='relu',padding='same',kernel_initializer='normal'))

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(640, kernel_initializer='normal', activation='sigmoid'))

    model.add(Dense(10, kernel_initializer='normal', activation='softmax')) #numbers 0 to 9

    # Compile model

    opti = optimizers.adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

    return model
# Build & Fit the model



print("-----X_train.shape  --", X_train.shape, X_train.__sizeof__()) #(42000, 784) 263424080

print("-----y_train.shape  --", y_train.shape, y_train.size) #(42000, 10) 420000



X_train = np.asarray(X_train) #you can not just pass the csv to the model. it has to be an array

print("-----X_train.shape with array  --", X_train.shape, X_train.__sizeof__()) #(42000, 784) 112

X_train_CNN = X_train.reshape(-1, 28, 28, 1) #Gray impage = 1, 4D( n_images, x_shape, y_shape, n_steps )

# CNN expects 4D as an input

print("-----X_train_CNN.shape with array  --", X_train_CNN.shape, X_train_CNN.__sizeof__()) # (42000, 28, 28, 1) 144



# build the model

model = CNN_Model()



#Early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=9)



history = model.fit(X_train_CNN, y_train, validation_split=0.20, epochs=170, batch_size=64, verbose=2,

					callbacks=[es]) # 15% split: Train on 35700 samples, validate on 6300 samples


# plot the accuracy and loss

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.title('Plot History: Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='test loss')

plt.title('Plot History: Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
########################################

# MNIST Test Data

########################################

test_MNIST_X = pd.read_csv('../input/test.csv')



test_MNIST_X.head(10)
# plot some testing images, randomly selected



test_x = test_MNIST_X.values.reshape(-1,28,28)

test_x = test_x / 255.0

sample_list = [0, 1, 55, 99, 200, 215, 354, 375, 

               400, 425, 475, 500, 250, 450, 525, 600, 

               700, 35, 155, 175, 181, 23, 5, 69, 

               26, 7, 70, 27995, 27996, 27997, 27998, 27999,

              2, 4, 75, 225, 250, 515, 545, 625,

               1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,

              1100, 2200, 3300, 4400, 5500, 6600, 7700, 8800,

              9900, 10000, 11000, 12000, 13000, 14000, 15000, 16000]

itr = iter(sample_list)

plt.figure(figsize=(12,10))

x, y = 16, 4

j = 0

for i in itr:

    plt.subplot(y, x, j+1)

    plt.imshow(test_x[i].reshape((28,28)),interpolation='nearest')

    plt.title(str(j))

    j = j + 1

plt.show()
test_MNIST_X = np.asarray(test_MNIST_X) #(28000, 10)

test_MNIST_X = test_MNIST_X.reshape(-1, 28, 28, 1) # Conv2D model and 4D conversion



# compare with  predicted outcome

Y_pred = model.predict(test_MNIST_X, verbose=2)



pred = np.argmax(Y_pred, axis=1)



actual_numbers_test = [2, 0, 2, 4, 1, 2, 5, 6, # This is manually generated using above

                       9, 1, 0, 6, 9, 9, 1, 5, # plot function

                       4, 2, 3, 3, 8, 5, 7, 1, 

                       2, 3, 6, 9, 7, 3, 9, 2, 

                       9, 3, 8, 0, 9, 3, 3, 8,

                       0, 8, 6, 4, 1, 0, 8, 6, 

                       4, 1, 1, 9, 4, 1, 4, 7, 

                       8, 0, 5, 9, 8, 3, 2, 4]

print("----Compare our result---", len(actual_numbers_test))

Yes = 0

No = 0

for i in range(pred.size):

	if(actual_numbers_test[i] == pred[sample_list[i]]):

		print("Index... Got it! Actual = , Predicted = ", i, actual_numbers_test[i],pred[sample_list[i]])

		Yes = Yes + 1

	else:

		print("Index... Nope....! (: Actual = , Predicted = ", i, actual_numbers_test[i], pred[sample_list[i]])

		No = No + 1

	if(i > (len(actual_numbers_test) - 2)):

		break



print("Correctly Predicted vs Error", Yes, No )

print("Succcess..", '%.2f' % ((Yes / len(actual_numbers_test)) * 100)+"%")
################################

# For the submission

################################



HV_submission = pd.DataFrame({'ImageId': range(1,len(test_MNIST_X)+1) ,'Label':pred })

#print(HV_submission.to_csv)



HV_submission.to_csv("hv_submission_23.csv",index=False)