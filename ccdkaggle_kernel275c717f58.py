# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing Libraries

import numpy as np 
import pandas as pd
import os
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt
from itertools import chain
from random import sample

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import keras
from sklearn.metrics import classification_report, confusion_matrix

# Deep learning libraries
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers, regularizers

train_data_dir = ('../input/chest-xray-pneumonia/chest_xray/train')
test_data_dir = ('../input/chest-xray-pneumonia/chest_xray/test')
val_data_dir = ('../input/chest-xray-pneumonia/chest_xray/val')
# Get all the data in the directory data/test , and reshape them
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_data_dir, 
        target_size=(150, 150), batch_size=624,class_mode='binary',seed=42)

# Get all the data in the directory data/train, and reshape them
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(train_data_dir, 
        target_size=(150, 150), batch_size=32,class_mode='binary',seed=42)

# Get all the data in the directory data/val, and reshape them
val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(val_data_dir, 
        target_size=(150, 150), batch_size=32,class_mode='binary',seed=42)

# Create the datasets
train_images, train_labels = next(train_generator)
test_images, test_labels = next(test_generator)
val_images, val_labels = next(val_generator)
input_path = '../input/chest-xray-pneumonia/chest_xray/'

# Distribution of our datasets
for _set in ['train', 'val', 'test']:
    n_normal = len(os.listdir(input_path + _set + '/NORMAL'))
    n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))
    print('In the Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
ax = ax.ravel()
plt.tight_layout()

for i, _set in enumerate(['train', 'val', 'test']):
    set_path = input_path+_set
    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')
    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
    ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')
    ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
print(np.shape(train_images))
print(np.shape(train_labels))
print(np.shape(test_images))
print(np.shape(test_labels))
np.unique(test_labels)
#array_to_img(test_images[7])
from IPython.display import Image
Image(width=128,height=128,filename='../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0115-0001.jpeg') 
Image(width=128,height=128,filename='../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1589_bacteria_4171.jpeg') 
#from keras.layers import MaxPooling2D
cnn = Sequential()

#Convolution
cnn.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(150, 150, 3)))

#Pooling
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd Convolution
cnn.add(Conv2D(32, kernel_size=(3, 3), activation="relu",kernel_regularizer=regularizers.l2(0.01)))

# 2nd Pooling layer
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the layer
cnn.add(Flatten())

# Fully Connected Layers
cnn.add(Dense(activation = 'relu', units = 512,kernel_regularizer=regularizers.l2(0.01)))
cnn.add(Dense(activation = 'sigmoid', units = 1))

# Compile the Neural network
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.summary()
cnn_history = cnn.fit_generator(train_generator,
                         steps_per_epoch = 80,
                         epochs = 10,
                         validation_data = test_generator)

# save time replace 
#                         steps_per_epoch = 80,
#                         epochs = 10,

# An epoch refers to one cycle through the full training dataset.An epoch is often mixed up with an iteration. 
# Iterations is the number of batches or steps through partitioned packets of the training data, needed to complete one epoch.
loss,test_accu = cnn.evaluate_generator(test_generator)
print('The testing accuracy is :',test_accu*100, '%')
print('The loss is :',loss*100, '%')
history_dict = cnn_history.history
history_dict.keys()
import matplotlib.pyplot as plt
import itertools
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2


def acc_loss_plot(model_his):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(model_his.history['acc'])
    ax[0].plot(model_his.history['val_acc'])
    ax[0].set(xlabel = 'Epoch', ylabel ='Accuracy' )
    ax[0].set_title('Model Accuracy', weight='bold')
    ax[0].legend(['Training set', 'Validation set'], loc='upper right')


    ax[1].plot(model_his.history['val_loss'],label='Val Loss')
    ax[1].plot(model_his.history['loss'],label='Loss')
    ax[1].set_title('Model Loss', weight='bold')
    ax[1].set(xlabel = 'Epoch', ylabel ='Loss' )
    ax[1].legend(loc='upper right')

    plt.show()
acc_loss_plot(cnn_history)
import datetime

original_start = datetime.datetime.now()
start = datetime.datetime.now()
start = datetime.datetime.now()
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_data_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),shuffle=True,
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary',seed=42)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_data_dir,
                 target_size=(150, 150), 
                 batch_size=32, 
                 class_mode='binary', 
                 shuffle=False,seed=42)

validation_generator = test_datagen.flow_from_directory(val_data_dir,
                       target_size=(150, 150),
                       batch_size=32,
                       class_mode='binary')


epochs = 10
batch_size = 32
#Earlier 2D convolutional layers, closer to the input, learn less filters, while later convolutional layers, closer to the 
#output, learn more filters. The number of filters you select should depend on the complexity of your dataset and the depth 
#of your neural network. A common setting to start with is [32, 64, 128] for three layers, and if there are more layers, 
#increasing to [256, 512]

model = models.Sequential()
model.add(layers.Conv2D(filters=8, kernel_size=(7, 7), activation='relu',padding='same',input_shape=(150, 150, 3)))
model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))

# Creating model and compiling
optimizer = Adam(lr=0.0001, decay=1e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
epochs = 10
batch_size = 32

# to safe time use below
#epochs = 3
#batch_size = 347

history = model.fit_generator(train_generator, 
                              steps_per_epoch=train_generator.samples // batch_size, 
                              epochs=epochs, 
                              validation_data=test_generator,callbacks=[checkpoint, lr_reduce],
                              validation_steps=test_generator.samples)
loss,test_accu = model.evaluate_generator(test_generator)
print('The testing accuracy is :',test_accu*100, '%')
print('The loss is :',loss*100, '%')
history_dict = history.history
history_dict.keys()
acc_loss_plot(history)
model.save('chest_xray_all_with_augmentation_data.h5')

end = datetime.datetime.now()
elapsed = end - start
print('Full data model training and evaluation took a total of:\n {}'.format(elapsed))
#test_generator = test_datagen.flow_from_directory(test_data_dir, 
#                                                  target_size=(150,150), 
#                                                  batch_size=32, 
#                                                  class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
end = datetime.datetime.now()
elapsed = end - original_start
print('Entire notebook took a total of:\n {}'.format(elapsed))
def show_eval(model, result, generator, labels):
    preds = model.predict_generator(generator)

    acc = accuracy_score(labels, np.round(preds))*100
    cm = confusion_matrix(labels, np.round(preds))
    tn, fp, fn, tp = cm.ravel()

    print('CONFUSION MATRIX ------------------')
    print(cm)

    print('\nTEST METRICS ----------------------')
    precision = (tp/(tp+fp))*100
    recall = (tp/(tp+fn))*100
    print('Accuracy: {}%'.format(acc))
    print('Precision: {}%'.format(precision))
    print('Recall: {}%'.format(recall))
    print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

    print('\nTRAIN METRIC ----------------------')
    print('Train acc: {}'.format(np.round((result.history['acc'][-1])*100, 2)))

    classes=np.unique(labels) 
    normalize=False
    title='Confusion matrix'
    cmap=plt.cm.Blues


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
    plt.show()
show_eval(model, new_history, test_generator, test_labels)
# get predictions on the test set
y_hat = model.predict(val_images)

img_labels = ['NORMAL', 'PNEUMONIA']
# Checking from Val data set as to how good our model is predicting
fig = plt.figure(figsize=(15, 8))
#fig.tight_layout(hspace=0.2,wspace=0.2)
plt.subplots_adjust(left=0.125,bottom=0.1, 
                    wspace=0.4, hspace=0.35)
for i, idx in enumerate(np.random.choice(val_images.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(val_images[idx]))
    pred_idx = int(np.round(y_hat[idx]))
    true_idx = int(np.round(val_labels[idx]))
    ax.set_title("{} \n{}".format(img_labels[pred_idx], img_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 16, figsize=(20,20))
    axes = axes.flatten() 
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
from itertools import chain

plotImages(val_images)
print(val_labels)
print(np.round(list(chain.from_iterable(y_hat))))
show_eval(model, new_history, val_generator, val_labels)