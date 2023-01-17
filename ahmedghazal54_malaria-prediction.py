import pandas as pd
import numpy as np
import os
import cv2
import itertools
#import imutils
import matplotlib.pyplot as plt
from keras.applications import VGG16
from sklearn.metrics import accuracy_score,confusion_matrix 
from sklearn.model_selection import train_test_split 
from keras.preprocessing import image
from keras.models import Model,Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop,Adam 
from keras.preprocessing.image import ImageDataGenerator
import shutil

BASE_DATASET_FOLDER = '/kaggle/input/files1/Malaria Cells'
VALIDATION_FOLDER = 'testing_set'
TRAIN_FOLDER = 'training_set'
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
IMAGE_SIZE = (100,100)
INPUT_SHAPE = (100,100, 3)
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0001
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DATASET_FOLDER, TRAIN_FOLDER),
        target_size=IMAGE_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='binary', 
        shuffle=True)
augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)
val_datagen=ImageDataGenerator(rescale=1./255)
val_generator=val_datagen.flow_from_directory(
    os.path.join(BASE_DATASET_FOLDER, VALIDATION_FOLDER),
    target_size=IMAGE_SIZE,
    class_mode='binary', 
    shuffle=False)
classes = {v: k for k, v in train_generator.class_indices.items()}
print(classes)
CATEGORIES = ["Parasitized","Uninfected"]
for category in CATEGORIES:  
    path = os.path.join('/kaggle/input/files1/Malaria Cells/training_set',category)  
    x=0
    for img in os.listdir(path): 
        x+=1
        img_array = cv2.imread(os.path.join(path,img))  
        plt.imshow(img_array)  #r graph it
        plt.show()  # display!
        if x==10 : 
            break
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

for layer in vgg_model.layers[:-4]:
    layer.trainable = False
model=Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(.2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(.2))
model.add(Conv2D(200, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(.2))
model.add(Conv2D(265, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc', 
    mode='max',
    patience=6
)
history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples//val_generator.batch_size,
        callbacks=[es]
)
plt.figure(figsize=[14,10])
plt.subplot(211)
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[14,10])
plt.subplot(212)
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
import numpy as np

test_image = image.load_img('/kaggle/input/files1/Malaria Cells/single_prediction/Parasitised.png', target_size = (100,100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(train_generator.class_indices)
if result[0][0] == 1:
    prediction = 'Uninfected'
else:
    prediction = 'Parasitised'
print(prediction)