# Copyrights by Radu Dogaru & Ioana Dogaru 
# LightWeight CNN (L-CNN)
# No hidden Layers involved
# L-CNN is used for mobile computing platforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
import numpy as np # linear algebra
import keras.backend as K 
import time as ti 
import cv2
import os
import glob # for including images
import scipy.io as sio
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D  
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop, SGD, Adadelta, Adam 
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
mar = cv2.imread('../input/fruits/fruits-360/Training/Apple Red 2/109_100.jpg')
mar = cv2.cvtColor(mar,cv2.COLOR_BGR2RGB)
plt.imshow(mar)
print(type(mar))
print(mar.shape)
# Training dataset
# We will need the images in a 32x32x3 input format.

X_train = [] # training fruit images
y_train = [] # training fruit labels 

for dir_path in glob.glob("../input/fruits/fruits-360/Training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_train.append(img)
        y_train.append(img_label)
        
X_train=np.array(X_train)
y_train=np.array(y_train)
# Test dataset 
# Images will also be in a 32x32x3 format.

X_test = [] # test fruit images
y_test = [] # test fruit labels 

for dir_path in glob.glob("../input/fruits/fruits-360/Test/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test.append(img)
        y_test.append(img_label)

X_test=np.array(X_test)
y_test=np.array(y_test)
fig = plt.figure(figsize =(30,5))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(X_train[i]))
plt.imshow(X_train[0])
print((X_train[0]))
#We will rescale the pixels from 0-250 to 0-1 for a better performance of the model

X_train = X_train/255.0
X_test = X_test/255.0
plt.imshow(X_train[10])
print(X_train[0])
plt.imshow(X_test[0])
print(len(np.unique(y_train)))
print(len(np.unique(y_test)))
# Now we need to have them labeled with numbers from 0 - 131 
label_to_id={v:k for k, v in enumerate(np.unique(y_train))}
#print(label_to_id)

y_train_label_id = np.array([label_to_id[i] for i in y_train])
y_test_label_id = np.array([label_to_id[i] for i in y_test])
#We need to translate this to be "one hot encoded" so our CNN can understand, 
# otherwise it will think this is some sort of regression problem on a continuous axis

from keras.utils.np_utils import to_categorical
print(y_train_label_id.shape)

y_cat_train_label_id=to_categorical(y_train_label_id)
y_cat_test_label_id=to_categorical(y_test_label_id)

print(y_cat_train_label_id.shape)
print(y_cat_train_label_id[0])
print(X_train.shape)
print(X_test.shape)
print(y_cat_train_label_id.shape)
print(y_cat_test_label_id.shape)
model = Sequential()
nf1=20; nf2=40; nf3=20; 
pad='same'
    
# First conv layer
model.add(Conv2D(filters=nf1,padding=pad, kernel_size=(3,3),input_shape=(32,32,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(4, 4),strides=(2,2),padding=pad))
    
# Second conv layer
model.add(Conv2D(filters=nf2,padding=pad, kernel_size=(3,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(4, 4),strides=(2,2),padding=pad))
    
# Third conv layer 
model.add(Conv2D(filters=nf3,padding=pad, kernel_size=(3,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(4, 4),strides=(2,2),padding=pad))
 
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
    
# Output with 131 neurons ( 131 classes )
model.add(Dense(131, activation='softmax'))
    
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 204
input_shape=(32,32,3)
epochs=30



filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

result = model.fit(X_train,y_cat_train_label_id,
                       batch_size=15,
                       epochs=30,
                       verbose=1,
                       validation_data=(X_test,y_cat_test_label_id),
                       callbacks=callbacks_list
                      )
plt.figure(1)  
plt.plot(result.history['accuracy'])  
plt.plot(result.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
plt.plot(result.history['loss'])  
plt.plot(result.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
# Load weights
model.load_weights("weights-improvement-23-0.98.hdf5")

# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
print("Created model and loaded weights from file")
model.evaluate(X_test,y_cat_test_label_id)
model.metrics_names
from sklearn.metrics import classification_report, confusion_matrix


predictions=model.predict_classes(X_test)
print(classification_report(y_test_label_id,predictions))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
print('Confusion Matrix \n')

array=confusion_matrix(y_test_label_id,predictions)
print(array)
#model.save('L-CNN v4.0.h5')
print(label_to_id)