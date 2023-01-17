# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras # for neural networks models



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Fetch data

data = pd.read_csv('../input/train.csv')

data.head()
# define variables

y = keras.utils.to_categorical(data.label.values, num_classes=10) #One-hot encoding

X = data.drop(['label'], axis=1).values # numpy array where each row is a flatten (vector) image

X = X.reshape((X.shape[0], 28, 28, 1)) # numpy array where each row is a matrix image

X = X/255 # values ranging from 0 (white) to 1 (black)
# Split dataset into training set and test set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Imports

from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam
# Define convolutional model

def cnn_model():

    # input: 28x28 images with one channel -> (28, 28, 1) tensors.

    model = Sequential()



    # this applies 32 convolution filters of size 3x3 each.

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(300, activation='relu'))

    model.add(Dense(10, activation='softmax'))



    adam = Adam(lr=0.001, decay=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



    return model
# plot model summary

model = cnn_model()

model.summary()
batch_size = 64

epochs = 10



if (True):

    model = cnn_model()

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    loss_and_metric = model.evaluate(X_test, y_test, batch_size=64)

    print('Loss', loss_and_metric[0])

    print('Accuracy:', loss_and_metric[1])
import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score



y_true = np.argmax(y_test, axis=1)

y_pred = np.argmax(model.predict(X_test), axis=1)



# Plot confusion matrix

matrix = confusion_matrix(y_true, y_pred)

sn.heatmap(matrix, annot=True, fmt='d')



# For seeing where our algorithm is misclassifying numbers

row_sums = matrix.sum(axis=1, keepdims=True)

norm_confusion_matrix = matrix / row_sums

np.fill_diagonal(norm_confusion_matrix, 0)

plt.matshow(norm_confusion_matrix, cmap=plt.cm.gray)

plt.show()
# See in which numbers our algorithm is making mistakes.

# for example, our algorithm is missclasifying 6 with 9.

# Code from: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6



# Errors are difference between predicted labels and true labels

errors = (y_pred - y_true != 0)



Y_pred_classes_errors = y_pred[errors]

Y_pred_errors = model.predict(X_test)[errors]

Y_true_errors = y_true[errors]

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

# Training model with data augmentation

# Code from: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

    

idg.fit(X_train)
# Train a new model

batch_size = 64

epochs = 10

new_model = cnn_model()

history = model.fit_generator(idg.flow(X_train, y_train, batch_size=batch_size),

                              epochs=epochs, validation_data=(X_test, y_test))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Read test data

X_test_for_predictions = pd.read_csv('../input/test.csv')

X_test_for_predictions = X_test_for_predictions.values.reshape((X_test_for_predictions.shape[0], 28, 28, 1))

X_test_for_predictions = X_test_for_predictions / 255

predictions = np.argmax(new_model.predict(X_test_for_predictions), axis=1)
idx = np.arange(predictions.size).reshape(-1,1) + 1

output = np.hstack((idx, predictions.reshape(-1, 1))).astype(np.int)

np.savetxt('predictions.csv', output, fmt='%i', delimiter=",", header='ImageId,Label', comments='')