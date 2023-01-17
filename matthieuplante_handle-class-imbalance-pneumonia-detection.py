import numpy as np

import pandas as pd

import os

import cv2



import seaborn as sns

import matplotlib.pyplot as plt

import itertools    



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils import np_utils

from keras.models import Sequential , Model

from keras.layers import Dense , Dropout , Flatten , Input , Conv2D , MaxPool2D

from keras.optimizers import Adam

from keras.optimizers.schedules import ExponentialDecay

from keras.initializers import Constant
img_size = 150

labels = ["PNEUMONIA", "NORMAL"]



def loadData(type_dir):

    X = []

    Y = []

    for label in labels :

        path1 = os.path.join(maindir, type_dir)

        path = os.path.join(path1, label)

        class_num = labels.index(label)

        for img in os.listdir(path) :

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

                X.append(resized_arr) #RGB dimension like

                Y.append(class_num)

            except Exception as e:

                print(e)

    return np.array(X), np.array(Y)
maindir = "../input/chest-xray-pneumonia/chest_xray"



train_X, train_Y = loadData("train")

valid_X, valid_Y = loadData("val")

test_X, test_Y = loadData("test")
all_X = np.concatenate((train_X, valid_X, test_X))

all_Y = np.concatenate((train_Y, valid_Y, test_Y))
maindir = "../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19"



train_X, train_Y = loadData("train")

test_X, test_Y = loadData("test")
all_X = np.concatenate((all_X, train_X, test_X))

all_Y = np.concatenate((all_Y, train_Y, test_Y))
maindir = "../input/chest-xray-covid19-pneumonia/Data"



train_X, train_Y = loadData("train")

test_X, test_Y = loadData("test")
all_X = np.concatenate((all_X, train_X, test_X))

all_Y = np.concatenate((all_Y, train_Y, test_Y))
all_X.shape
train_X, valid_X, train_Y, valid_Y = train_test_split(all_X, all_Y, test_size = 0.1)
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size = 10/9 - 1)
train_X = np.array(train_X) / 255

valid_X = np.array(valid_X) / 255

test_X = np.array(test_X) / 255
train_X = train_X.reshape(-1, img_size, img_size, 1)

valid_X = valid_X.reshape(-1, img_size, img_size, 1)

test_X = test_X.reshape(-1, img_size, img_size, 1)
# Create the model

model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Valid', activation='relu', input_shape=(img_size, img_size, 1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



# Add new layers

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 1 , activation = 'sigmoid'))

opt = Adam(learning_rate=0.001)

model.compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])



# Show a summary of the model. Check the number of trainable parameters

model.summary()
history = model.fit(train_X, train_Y, batch_size = 218, epochs=21, validation_data = (valid_X, valid_Y))
epochs = [i for i in range(21)]

fig , ax = plt.subplots(1,2)

train_acc = history.history['accuracy']

train_loss = history.history['loss']

val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']

fig.set_size_inches(20,10)



ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')

ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')

ax[0].set_title('Training & Validation Accuracy')

ax[0].legend()

ax[0].set_xlabel("Epochs")

ax[0].set_ylabel("Accuracy")



ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')

ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')

ax[1].set_title('Testing Accuracy & Loss')

ax[1].legend()

ax[1].set_xlabel("Epochs")

ax[1].set_ylabel("Training & Validation Loss")

plt.show()
test = model.evaluate(test_X,test_Y)



print("Loss of the model on test distrib - " , test[0])

print("Accuracy of the model on test distrib - " , test[1]*100 , "%")
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

Y_pred = model.predict(test_X)

Y_pred = Y_pred.reshape(Y_pred.shape[0])

Y_pred = np.round(Y_pred)

test_Y = test_Y.astype(np.float32)

print(Y_pred, test_Y)



# compute the confusion matrix

confusion_mtx = confusion_matrix(test_Y, Y_pred) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(1)) 