import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import shutil
import cv2

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, LeakyReLU, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
 
%matplotlib inline
images=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images")
annotations=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations")
train=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/train.csv"))
submission=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/submission.csv"))
len(os.listdir(images))
!mkdir /kaggle/working/images
!mkdir /kaggle/working/images/train
!mkdir /kaggle/working/images/validation
!mkdir /kaggle/working/images/train/face_with_mask
!mkdir /kaggle/working/images/train/face_no_mask
!mkdir /kaggle/working/images/validation/face_with_mask
!mkdir /kaggle/working/images/validation/face_no_mask
source="/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/"
train="/kaggle/working/images/train/face_with_mask/"
train_1="/kaggle/working/images/train/face_no_mask/"

test="/kaggle/working/images/validation/face_with_mask/"
test_1="/kaggle/working/images/validation/face_no_mask/"

count=0
with open('/kaggle/input/face-mask-detection-dataset/train.csv') as csvfile:
    readCSV = list(csv.reader(csvfile, delimiter=','))
    print(len([row for row in readCSV[1:] if(row[5]=="face_with_mask" or row[5]=="face_no_mask")]))
    len_train_samples=int(len([row for row in readCSV[1:] if(row[5]=="face_with_mask" or row[5]=="face_no_mask")])*0.7)
    for row in readCSV[1:]:
        if(row[5]=="face_with_mask" or row[5]=="face_no_mask"):
            count+=1
            x1=int(row[1])
            x2=int(row[2])
            y1=int(row[3])
            y2=int(row[4])
            
            image=cv2.imread(src_path+row[0])
            image=image[x2:y2,x1:y1]
            
            if(count<=len_train_samples and row[5]=="face_with_mask"):
                cv2.imwrite(train+str(count)+".jpg",image)
            
            elif(count<=len_train_samples and row[5]=="face_no_mask"):
                cv2.imwrite(train_1+str(count)+".jpg",image)
            
            elif(count>len_train_samples and row[5]=="face_with_mask"):
                cv2.imwrite(test+str(count)+".jpg",image)
            
            elif(count>len_train_samples and row[5]=="face_no_mask"):
                cv2.imwrite(test_1+str(count)+".jpg",image)
for pic in os.listdir("/kaggle/working/images/train/face_no_mask/")[0:5]:
    print(pic)
    img=plt.imread("/kaggle/working/images/train/face_no_mask/"+pic)
    plt.imshow(img)
len(os.listdir("/kaggle/working/images/train/face_no_mask/")) + len(os.listdir("/kaggle/working/images/train/face_with_mask/"))
len(os.listdir("/kaggle/working/images/validation/face_no_mask/")) + len(os.listdir("/kaggle/working/images/validation/face_with_mask/"))
4024+1725
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
training_dir = "/kaggle/working/images/train/"
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(training_dir, batch_size=5, class_mode='binary', target_size=(150, 150))

validation_dir = "/kaggle/working/images/validation/"
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=5, class_mode='binary',target_size=(150, 150))
history = model.fit_generator(train_generator, epochs=20, verbose=1, validation_data=validation_generator)
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) 

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')
