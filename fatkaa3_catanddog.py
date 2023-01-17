# Kütüphaneleri yüklüyoruz.

# TensorFlow ve tf.keras
import tensorflow as tf
from tensorflow import keras

# Diğer kütüphaneler
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas as pd
import os
import random

import warnings
warnings.filterwarnings('ignore')
# Train ve test dataset okuma
data_test ='../input/test_set/test_set/'
data_train ='../input/training_set/training_set/'

folders_train = os.listdir(data_train)
print(folders_train)

folders_test = os.listdir(data_test)
print(folders_test)
# Train ve Test tek Datasette birleştirildi

image_names = []
data_labels = []
data_images = []

size = 64,64
def dataset_birlestirme(folders, data):
    for folder in folders:
        for file in os.listdir(os.path.join(data,folder)):
            if file.endswith("jpg"):
                image_names.append(os.path.join(data,folder,file))
                data_labels.append(folder)
                img = cv2.imread(os.path.join(data,folder,file))
                im = cv2.resize(img,size)
                data_images.append(im)
            else:
                continue


dataset_birlestirme(folders_train,data_train)     
dataset_birlestirme(folders_test,data_test)

# dataset kontrolü
data = np.array(data_images)
data.shape
# Labellar kategorik hale getirildi (cat, dog)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

le = LabelEncoder()
data_encod = le.fit_transform(data_labels)
data_ohe = np_utils.to_categorical(data_encod)

# Label kategorik hali
print(pd.unique(data_labels))
# Label encode hali
print(pd.unique(data_encod))

# Data içindeki örnek kontrolü
plt.imshow(data[4515][:,:,0],cmap='gray')
plt.title(data_labels[4501])
plt.show()
# Datasetimizdeki örnekler
import warnings
warnings.filterwarnings('ignore')

_, ax = plt.subplots(5,6, figsize=(30,30))
for i in range(5):
    for j in range(6):
      ax[i,j].imshow(cv2.cvtColor(data[100+(i*1000)+j], cv2.COLOR_BGR2RGB))
      ax[i,j].axis('off')
      ax[i,j].set_title(le.inverse_transform(data_encod[100+(i*1000)+j]))
plt.show()
# Train ve validation datasetler ayrıldı
from sklearn.cross_validation import train_test_split

X_train,X_val,y_train,y_val=train_test_split(data,data_encod,test_size=0.20,random_state=42)

print("X_train shape",X_train.shape)
print("X_test shape",X_val.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_val.shape)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same',
                             input_shape=(64,64,3), activation='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', 
                        activation='relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 
                        activation='relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', 
                        activation='relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 
                        activation='relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(output_dim=64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
from keras.preprocessing.image import ImageDataGenerator
# Veri Çoğaltma
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.5,
        zoom_range = 0.5,
        width_shift_range=0.5,
        height_shift_range=0.5,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(X_train)
epochs = 50
batch_size = 150

# Modeli Eğitme
history = model.fit_generator(
    datagen.flow(X_train,y_train, batch_size=batch_size),
    epochs = epochs,
    validation_data = (X_val,y_val), 
    steps_per_epoch=X_train.shape[0] // batch_size)
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(history.history['acc'], color = 'blue')
axarr[0].plot(history.history['val_acc'], color = 'purple')
axarr[0].legend(['train', 'test'])
axarr[0].set_title('acc - val_acc')
axarr[1].plot(history.history['loss'], color = 'red')
axarr[1].plot(history.history['val_loss'], color = 'gray')
axarr[1].legend(['train', 'test'])
axarr[1].set_title('loss - val_loss')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

pred = model.predict_classes(X_val)
cm = confusion_matrix(y_val,pred)

f,ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Purples",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import numpy as np
from keras.preprocessing import image


test_image=image.load_img('../input/test_set/test_set/dogs/dog.4006.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict_classes(test_image)


if result[0][0] >=0.5:
    prediction='dog'
else:
    prediction='cat'
print(prediction)