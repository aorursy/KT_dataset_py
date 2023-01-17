#import necessary files
import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
print(os.listdir("../input"))
#import necessary files
import os
import sys
import cv2
from keras.utils import to_categorical
import matplotlib
from keras import backend as k
k.clear_session()

#Lists for training data
X_train = []
y_train = []

#Assigning numbers to labels
labels = { "Black-grass":0,"Charlock":1,"Cleavers":2,"Common Chickweed":3,"Common wheat":4,"Fat Hen":5,"Loose Silky-bent":6,"Maize":7,
"Scentless Mayweed":8,"Shepherds Purse":9,"Small-flowered Cranesbill" :10,"Sugar beet":11}
#to show progress bars for loops
from tqdm import tqdm
labels_list = []
#Size of image
width = 128

#Getting files and seperating it into X and y
directories = os.listdir("../input/plant-seedlings-classification/train/")
for k in tqdm(range(len(directories))):   
    files = os.listdir("../input/plant-seedlings-classification/train/{}".format(directories[k]))
    for f in range(len(files)):    
        img = cv2.imread('../input/plant-seedlings-classification/train/{}/{}'.format(directories[k], files[f]))
        target_list = np.zeros(12)
        labels_list.append(labels[directories[k]])
        target_list[labels[directories[k]]] = 1 
        X_train.append(cv2.resize(img, (width, width)))
        y_train.append(target_list)
    
y_train = np.array(y_train, np.uint8)
X_train = np.array(X_train, np.float32)

print(X_train.shape)
print(y_train.shape)
from matplotlib import pyplot as plt
#Plotting histogram of number of different species
plt.hist(labels_list)
plt.title('Frequency Histogram of Species')
plt.figure(figsize=(12, 12))
plt.show()
#checking to see if the input was correct
X_train[0]
#Checking how the sample images look
%matplotlib inline
import os
import matplotlib
import matplotlib.pyplot as plt
for i in range(1,7):
    plt.figure(figsize=(10,10))
    ax = plt.subplot(3, 3, i + 1)
    new_image = tf.keras.preprocessing.image.array_to_img(X_train[i])
    plt.imshow(new_image)
    plt.show()


#Using shuffle split to shuffle the data and slpit it
from sklearn.model_selection import StratifiedShuffleSplit
#Dividing all values of X_train by 255 to get the values between 0 and 1 
X_train/=255

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=30) 
for train_index, test_index in sss.split(X_train, y_train):
    print("{} are used for training and {} are used for testing".format(len(train_index), len(test_index)))
    X_train, X_valid = X_train[train_index], X_train[test_index]
    y_train, y_valid = y_train[train_index], y_train[test_index]
X_train
#Using keras ImageDataGenerator to generate multiple images from single image
from keras.preprocessing.image import ImageDataGenerator
dataGen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")
print(X_train)

from keras import backend as k
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import Adam
# initialize the model

sys.stdout.flush()
k.clear_session()

epochs = 50
INIT_LR = 1e-3
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size= (3, 3), padding="same", input_shape=(128,128,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 64,kernel_size = (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=600))
model.add(Activation("relu"))

# Classifying using softmax
model.add(Dense(units=12))
model.add(Activation("softmax"))
   
# opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001, decay=0.001/50), metrics=["accuracy"])
model.summary()

# training the model with 50 epochs
sys.stdout.flush()
cnn_model_50 = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=32), 
                        validation_data=(X_valid, y_valid), 
                        steps_per_epoch=len(X_train)/32, 
                        epochs=200, verbose=1)

val_pred = model.predict(x=X_valid,verbose=1)
y_pred = [np.argmax(probas) for probas in val_pred]
y_pred = np.argmax(val_pred,axis=1)
y_pred.shape
y_test = np.argmax(y_valid,axis=1)
y_test.shape
from sklearn.metrics import confusion_matrix
class_names = ["Black-grass","Charlock","Cleavers","Common Chickweed","Common wheat","Fat Hen","Loose Silky-bent","Maize","Scentless Mayweed","Shepherds Purse",
               "Small-flowered Cranesbill","Sugar beet"]
import itertools

def plot_confusion_matrix(cm,classes,title='Confusion Matrix',cmap=plt.cm.Blues):
    
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
        pass
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    pass

cnf_mat = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)


plt.figure()
plot_confusion_matrix(cnf_mat,classes=class_names)
plt.show()
#plotting graph for Training and Validation accuracy
def plot_hist(cnn_model):
    plt.plot(cnn_model_50.history["acc"],label="Training")
    plt.plot(cnn_model_50.history["val_acc"],label="Validation")
    plt.title("Model training and validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()
    

plot_hist(cnn_model_50)
#Plotting graph for training and validation loss 
def plot_hist_loss(cnn_model):
    plt.plot(cnn_model_50.history["loss"],label="Training")
    plt.plot(cnn_model_50.history["val_loss"],label="Validation")
    plt.title("Model training and validation loss")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()
    

plot_hist_loss(cnn_model_50)
val_acc = cnn_model_50.history["val_acc"]
train_acc = cnn_model_50.history["acc"]
val_loss = cnn_model_50.history["val_loss"]
train_loss = cnn_model_50.history["loss"]
len(val_acc)
max_train_acc_20e = max(train_acc[0:20])
max_val_acc_20e = max(val_acc[0:20])
min_train_loss_20e = min(train_loss[0:20])
min_val_loss_20e = min(val_loss[0:20])
max_train_acc_30e = max(train_acc[0:30])
max_val_acc_30e = max(val_acc[0:30])
min_train_loss_30e = min(train_loss[0:30])
min_val_loss_30e = min(val_loss[0:20])
max_train_acc_40e = max(train_acc[0:40])
max_val_acc_40e = max(val_acc[0:40])
min_train_loss_40e = min(train_loss[0:40])
min_val_loss_40e = min(val_loss[0:20])
max_train_acc_50e = max(train_acc[0:50])
max_val_acc_50e = max(val_acc[0:50])
min_train_loss_50e = min(train_loss[0:50])
min_val_loss_50e = min(val_loss[0:20])
max_train_acc_100e = max(train_acc[0:100])
max_val_acc_100e = max(val_acc[0:100])
min_train_loss_100e = min(train_loss[0:100])
min_val_loss_100e = min(val_loss[0:100])
max_train_acc_200e = max(train_acc[0:200])
max_val_acc_200e = max(val_acc[0:200])
min_train_loss_200e = min(train_loss[0:200])
min_val_loss_200e = min(val_loss[0:200])
print(max_train_acc_20e,
max_val_acc_20e,
min_train_loss_20e,
min_val_loss_20e,
max_train_acc_30e,
max_val_acc_30e,
min_train_loss_30e,
min_val_loss_30e,
max_train_acc_40e,
max_val_acc_40e,
min_train_loss_40e,
min_val_loss_40e,
max_train_acc_50e,
max_val_acc_50e,
min_train_loss_50e,
min_val_loss_50e,
max_train_acc_100e,
max_val_acc_100e,
min_train_loss_100e,
min_val_loss_100e,
max_train_acc_200e,
max_val_acc_200e,
min_train_loss_200e,
min_val_loss_200e      
)
