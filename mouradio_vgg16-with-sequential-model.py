# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt
import seaborn as sns

import os, zipfile, random

# Import the backend
import tensorflow as tf

# Data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# Model architecture
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Model evaluation
from sklearn.metrics import accuracy_score
import os, cv2, re, random

img_size = (150, 150)
input_dim = (150, 150, 3)
epochs = 30
batch_size = 16
local_zip_train = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
local_zip_test = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
zip_ref = zipfile.ZipFile(local_zip_train, 'r')
zip_ref.extractall('/kaggle/working')
zip_ref.close()
filenames = os.listdir("/kaggle/working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()
images_list = []
for i in df['filename']:
    i='/kaggle/working/train/'+i
    images_list.append(i)
img_width = 150
img_height = 150
def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = [] # images as arrays
    y = [] # labels
    
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
    
    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        #else:
            #print('neither cat nor dog name present in images')
            
    return x, y
X,Y = prepare_data(images_list) 
import random as rn

fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Y))
        ax[i,j].imshow(X[l])
       # ax[i,j].set_title('Dog: '+Y[l])
        
plt.tight_layout()
        


X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras import layers, models, optimizers
model=models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])




train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)




train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)


nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16


visualize_acc_loss = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=40,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)




fig, ax = plt.subplots(2,1)
ax[0].plot(visualize_acc_loss.history['loss'], color='b', label="Training loss")
ax[0].plot(visualize_acc_loss.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(visualize_acc_loss.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(visualize_acc_loss.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


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

from sklearn.metrics import confusion_matrix
import itertools

Y_pred = model.predict(np.array(X_val))

Y_pred_labels = [] # labels    
    
for i in range(0,len(Y_pred)):
    if Y_pred[i, 0] >= 0.5: 
        Y_pred_labels.append(1)
    else:
        Y_pred_labels.append(0)

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_val, Y_pred_labels) 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(1))
Y_pred = model.predict(np.array(X_val))

#####predict cat | predict dog
for i in range(0,5):
    if Y_pred[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(Y_pred[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-Y_pred[i][0]))
        
    plt.imshow(X_val[i])    
    plt.show()
def prepare_data_test(list_of_images):

    x = [] # images as arrays
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (150,150), interpolation=cv2.INTER_CUBIC))
        return x
local_zip_test = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
zip_ref = zipfile.ZipFile(local_zip_test, 'r')
zip_ref.extractall('/kaggle/working')
zip_ref.close()
filenames = os.listdir("/kaggle/working/test")
categories = []
for filename in filenames:
    categories.append(filename)
images_list = []
for i in categories:
    i='/kaggle/working/test/'+i
    images_list.append(i)
images_list
test_image=prepare_data_test(images_list)
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_generator = test_datagen.flow(np.array(test_image), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1,steps = len(test_generator))
counter = range(1, len(images_list) + 1)
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols =solution.columns

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("dogsVScats.csv", index = False)
