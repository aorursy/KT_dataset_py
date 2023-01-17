import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_curve,auc,classification_report

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout  

from keras.callbacks import EarlyStopping, ModelCheckpoint

from random import shuffle

from tqdm import tqdm  

import scipy

import skimage

from skimage.transform import resize

import random

import os

print(os.listdir("../input"))

# setting path of directory

PARA_DIR = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"

NORM_DIR =  "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"



# storing all the files from directories PARA_DIR and NORM_DIR to Pimages and Nimages for accessing images directly

Pimages = os.listdir(PARA_DIR)

Nimages = os.listdir(NORM_DIR)
sample_parasite = random.sample(Pimages,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/'+sample_parasite[i])

    ax[i//3,i%3].imshow(im)

    ax[i//3,i%3].axis('off')

f.suptitle('Parasite infected blood sample images',fontsize=20)

plt.show()
sample_normal = random.sample(Nimages,6)

f,ax = plt.subplots(2,3,figsize=(15,9))



for i in range(0,6):

    im = cv2.imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/'+sample_normal[i])

    ax[i//3,i%3].imshow(im)

    ax[i//3,i%3].axis('off')

f.suptitle('Normal Blood Sample Images (Un-infected)', fontsize=20)

plt.show()
data=[]

labels=[]

Parasitized=os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/")

for a in Parasitized:

    try:

        image=cv2.imread("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+a)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")



Uninfected=os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/")

for b in Uninfected:

    try:

        image=cv2.imread("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+b)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")
# segregating data and labels

Cells=np.array(data)

labels=np.array(labels)



np.save("Cells",Cells)

np.save("labels",labels)
# loading data of cell images and labels of images

Cells=np.load("Cells.npy")

labels=np.load("labels.npy")
s=np.arange(Cells.shape[0])

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]



num_classes=len(np.unique(labels))

len_data=len(Cells)
# splitting cells images into 90:10 ratio i.e., 90% for training and 10% for testing purpose

(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]



(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.

x_test = x_test.astype('float32')/255

train_len=len(x_train)

test_len=len(x_test)
#Doing One hot encoding as classifier has multiple classes

y_train=keras.utils.to_categorical(y_train,num_classes)

y_test=keras.utils.to_categorical(y_test,num_classes)
# Set random seed

np.random.seed(0)



#creating sequential model

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))





model.add(Flatten())



model.add(Dense(512,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint('.mdl_wts.hdf5', monitor='val_loss', save_best_only=True)]
#Fit the model with min batch size as 32 can tune batch size to some factor of 2^power ] 

h=model.fit(x_train,y_train,batch_size=32,callbacks=callbacks, validation_data=(x_test,y_test),epochs=20,verbose=1)
# saving the weight of model

from numpy import loadtxt

from keras.models import load_model

model = load_model('.mdl_wts.hdf5')



#checking the score of the model

score=model.evaluate(x_test,y_test)

print(score)
# checking the accuracy of thr 

accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])
from sklearn.metrics import confusion_matrix

pred = model.predict(x_test)

pred = np.argmax(pred,axis = 1) 

y_true = np.argmax(y_test,axis = 1)



#creating confusion matrix

CM = confusion_matrix(y_true, pred)

from mlxtend.plotting import plot_confusion_matrix

# plotting confusion matrix

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))

plt.show()
def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])

    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()
plot_model_history(h)
print('{}'.format( 

                           classification_report(y_true , pred)))
fpr_keras, tpr_keras, thresholds = roc_curve(y_true.ravel(), pred.ravel())

auc_keras = auc(fpr_keras, tpr_keras)

auc_keras
def plot_roc_curve(fpr, tpr):

    plt.figure(figsize=(10,6))

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
plot_roc_curve(fpr_keras, tpr_keras)
y_hat = model.predict(x_test)



# define text labels 

malaria_labels = ['Parasitized','Uninfected']
# plot a random sample of test images, their predicted labels, and ground truth

fig = plt.figure(figsize=(20, 8))

for i, idx in enumerate(np.random.choice(x_test.shape[0], size=12, replace=False)):

    ax = fig.add_subplot(4,4, i+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[idx]))

    pred_idx = np.argmax(y_hat[idx])

    true_idx = np.argmax(y_test[idx])

    ax.set_title("{} ({})".format(malaria_labels[pred_idx], malaria_labels[true_idx]),

                 color=("blue" if pred_idx == true_idx else "orange"))