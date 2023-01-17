import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import seaborn as sns
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,img_to_array
data = pd.read_excel('../input/covidct/COVID-CT-MetaInfo.xlsx')
data.head()
covid_dir = os.listdir('/kaggle/input/covidct/CT_COVID/')
noncovid_dir = os.listdir('/kaggle/input/covidct/CT_NonCOVID/')
covid_dir[0]
print("Covid images: ",len(covid_dir))
print("Non-Covid images: ",len(noncovid_dir))
plt.style.use('dark_background')
covid_img = cv2.imread('../input/covidct/CT_COVID/2020.01.24.919183-p27-135.png')
non_covid_img = cv2.imread('../input/covidct/CT_NonCOVID/10%2.jpg')

fig = plt.figure(figsize=(10,10))
fig.add_subplot(1,2,1)
plt.imshow(covid_img)
plt.title('Covid image')

fig.add_subplot(1,2,2)
plt.imshow(non_covid_img)
plt.title('non covid image')
print("Covid image shape: {}".format(covid_img.shape))
print("Non Covid image shape: {}".format(non_covid_img.shape))
img_height, img_width = 228, 228
batch_size = 128
DIR = '/kaggle/input/covidct/'
DIR
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    validation_split=0.2) 

train_generator = train_datagen.flow_from_directory(
    DIR,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode="grayscale",
    subset='training') 

validation_generator = train_datagen.flow_from_directory(
    DIR, 
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode="grayscale",
    subset='validation') 
model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(img_height,img_height,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit_generator(train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = 50)
plt.style.use('seaborn')
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.lineplot(x=np.arange(1, 51), y=history.history.get('loss'), ax=ax[0, 0])
sns.lineplot(x=np.arange(1, 51), y=history.history.get('accuracy'), ax=ax[0, 1])
sns.lineplot(x=np.arange(1, 51), y=history.history.get('val_loss'), ax=ax[1, 0])
sns.lineplot(x=np.arange(1, 51), y=history.history.get('val_accuracy'), ax=ax[1, 1])
ax[0, 0].set_title('Training Loss vs Epochs')
ax[0, 1].set_title('Training AUC vs Epochs')
ax[1, 0].set_title('Validation Loss vs Epochs')
ax[1, 1].set_title('Validation AUC vs Epochs')
fig.suptitle('Using Adam Optimizer', size=16)
plt.show()
model1 = Sequential()
model1.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(img_height,img_height,1)))
model1.add(MaxPool2D(pool_size=(2,2)))

model1.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model1.add(MaxPool2D(pool_size=(2,2)))

model1.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model1.add(MaxPool2D(pool_size=(2,2)))


model1.add(Flatten())

model1.add(Dense(512,activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(256,activation='relu'))
model1.add(Dropout(0.3))

model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='RMSProp',metrics=['accuracy'])
history1 = model.fit_generator(train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = 50)
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.lineplot(x=np.arange(1, 51), y=history1.history.get('loss'), ax=ax[0, 0])
sns.lineplot(x=np.arange(1, 51), y=history1.history.get('accuracy'), ax=ax[0, 1])
sns.lineplot(x=np.arange(1, 51), y=history1.history.get('val_loss'), ax=ax[1, 0])
sns.lineplot(x=np.arange(1, 51), y=history1.history.get('val_accuracy'), ax=ax[1, 1])
ax[0, 0].set_title('Training Loss vs Epochs')
ax[0, 1].set_title('Training AUC vs Epochs')
ax[1, 0].set_title('Validation Loss vs Epochs')
ax[1, 1].set_title('Validation AUC vs Epochs')
fig.suptitle('Using RMSProp Optimizer', size=16)
plt.show()
