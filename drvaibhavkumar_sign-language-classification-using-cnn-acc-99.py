import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt 
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
print(train.shape)

print(test.shape)
train.head()
# Create training and testing arrays

train_set = np.array(train, dtype = 'float32')

test_set = np.array(test, dtype='float32')
#Specifying class labels

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]
#See a random image for class label verification

i = random.randint(1,27455)

plt.imshow(train_set[i,1:].reshape((28,28))) 



plt.imshow(train_set[i,1:].reshape((28,28))) 

label_index = train["label"][i]

plt.title(f"{class_names[label_index]}")

plt.axis('off')
# Define the dimensions of the plot grid 

W_grid = 5

L_grid = 5

fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_train = len(train_set) # get the length of the train dataset

# Select a random number from 0 to n_train

for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number

    index = np.random.randint(0, n_train)

    # read and display an image with the selected index    

    axes[i].imshow( train_set[index,1:].reshape((28,28)) )

    label_index = int(train_set[index,0])

    axes[i].set_title(class_names[label_index], fontsize = 8)

    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
# Prepare the training and testing dataset 

X_train = train_set[:, 1:] / 255

y_train = train_set[:, 0]



X_test = test_set[:, 1:] / 255

y_test = test_set[:,0]
#Visualize train images

plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i].reshape((28,28)), cmap=plt.cm.binary)

    label_index = int(y_train[i])

    plt.title(class_names[label_index])

plt.show()
#Split the training and test sets

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345)



print(X_train.shape)

print(y_train.shape)
# Unpack the training and test tuple

X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))

X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))



print(X_train.shape)

print(y_train.shape)

print(X_validate.shape)
#Library for CNN Model

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
#Defining the Convolutional Neural Network

cnn_model = Sequential()



cnn_model.add(Conv2D(32, (3, 3), input_shape = (28,28,1), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.25))



cnn_model.add(Conv2D(64, (3, 3), input_shape = (28,28,1), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.25))



cnn_model.add(Conv2D(128, (3, 3), input_shape = (28,28,1), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

cnn_model.add(Dropout(0.25))



cnn_model.add(Flatten())

cnn_model.add(Dense(units = 512, activation = 'relu'))

cnn_model.add(Dropout(0.25))

cnn_model.add(Dense(units = 25, activation = 'softmax'))



cnn_model.summary()
#Compiling

cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam' ,metrics =['accuracy'])
#Training the CNN model

history = cnn_model.fit(X_train, y_train, batch_size = 512, epochs = 50, verbose = 1, validation_data = (X_validate, y_validate))
#VIsualizing the training performance

plt.figure(figsize=(12, 8))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='val_Loss')

plt.legend()

plt.grid()

plt.title('Loss evolution')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label='val_accuracy')

plt.legend()

plt.grid()

plt.title('Accuracy evolution')
#Predictions for the test data

predicted_classes = cnn_model.predict_classes(X_test)
L = 5

W = 5

fig, axes = plt.subplots(L, W, figsize = (12,12))

axes = axes.ravel()



for i in np.arange(0, L * W):  

    axes[i].imshow(X_test[i].reshape(28,28))

    axes[i].set_title(f"Prediction Class = {predicted_classes[i]:0.1f}\n True Class = {y_test[i]:0.1f}")

    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
from sklearn.metrics import confusion_matrix

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predicted_classes)
#Defining function for confusion matrix plot

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Computing confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



# Visualizing

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



   # Rotating the tick labels and setting their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")

    # Looping over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

np.set_printoptions(precision=2)
#Specifying class labels

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]
plt.figure(figsize=(20,20))

plot_confusion_matrix(y_test, predicted_classes, classes = class_names, title='Non-Normalized Confusion matrix')

plt.show()
plt.figure(figsize=(35,35))

plot_confusion_matrix(y_test, predicted_classes, classes = class_names, normalize=True, title='Non-Normalized Confusion matrix')

plt.show()
#Classification accuracy

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test, predicted_classes)

print('Accuracy Score = ',acc_score)