import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.preprocessing import image
import PIL
%cd ../input/
os.listdir()
os.listdir('merged-trash-dataset/DATASET/')
print('TRAIN R IS ', len(os.listdir('merged-trash-dataset/DATASET/TRAIN/R')))
print('TRAIN O IS ', len(os.listdir('merged-trash-dataset/DATASET/TRAIN/O')))
print('TEST R IS ', len(os.listdir('merged-trash-dataset/DATASET/TEST/R')))
print('TEST O IS ', len(os.listdir('merged-trash-dataset/DATASET/TEST/O')))
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input, Dropout

model = Sequential()
model.add(Conv2D(224, kernel_size=(5, 5), input_shape=(224,224,3),activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
model.summary()
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=40,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True
                              )
test_gen = ImageDataGenerator(rescale=1./255)

train_dir = 'merged-trash-dataset/DATASET/TRAIN/'
test_dir = 'merged-trash-dataset/DATASET/TEST/'

train_generator = train_gen.flow_from_directory(train_dir, batch_size = 32, target_size = (224, 224), class_mode = 'binary')
test_generator = test_gen.flow_from_directory(test_dir, batch_size = 32, target_size = (224, 224), class_mode = 'binary')

print(train_generator)
print(test_generator)
hist = model.fit_generator(train_generator, epochs = 40, validation_data = test_generator)

%matplotlib inline
acc = hist.history['acc']
loss = hist.history['loss']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Testing Accuracy")
plt.title('Training vs Testing Accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Testing Loss")
plt.title('Training vs Testing Loss')
plt.show()
from sklearn.metrics import classification_report, confusion_matrix

pred = model.predict_generator(test_generator, verbose = 1)
y_pred = np.argmax(pred, axis = 1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes,y_pred))
print('Classification Report')
target_names = ['Organic', 'Recyclable']
print(classification_report(test_generator.classes,y_pred, target_names = target_names))


%cd ../working/ 
os.listdir() 
model.save('trashModel2.h5')
import cv2
video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = video_capture.read()
# Close device
video_capture.release()

import sys
from matplotlib import pyplot as plt
frameRGB = frame[:,:,::-1] # BGR => RGB
plt.imshow(frameRGB)
%cd ../input/predictiondata/
from PIL import Image
import numpy
org_img = Image("Organic.jpg")
rec_img = Image("Recyclable.jpg")
labelNames=['Organic','Recyclable']
#Organic prediction
org_img1 = numpy.array(org_img)
print('Actual label: Organic')
# Prepare image to predict
#test_image =np.expand_dims(org_img, axis=0)
#print('Input image shape:',org_img.shape)
print('Predict Label:',labelNames[model.predict_classes(org_img1,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(org_img1,batch_size=1))
#Recyclable prediction
plt.imshow(rec_img,aspect='auto')
print('Actual label: Recyclable')
# Prepare image to predict
test_image =np.expand_dims(rec_img, axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',labelNames[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))