# Simple implementation of the MobileNet v2.0 using Keras
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
from keras import layers
from keras import models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D  
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop, SGD, Adadelta, Adam 
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import MobileNetV2
X_train = [] # training fruit images
y_train = [] # training fruit labels 

X_test = [] # test fruit images
y_test = [] # test fruit labels 
# Training dataset
# We will need the images in a 32x32x3 input format.


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
X_train = X_train/255.0
X_test = X_test/255.0
print(y_test)
# Now we need to have them labeled with numbers from 0 - 120 
label_to_id={v:k for k, v in enumerate(np.unique(y_train))}
#print(label_to_id)

y_train_label_id = np.array([label_to_id[i] for i in y_train])
y_test_label_id = np.array([label_to_id[i] for i in y_test])

# We need to translate this to be "one hot encoded" so our CNN can understand, 
# otherwise it will think this is some sort of regression problem on a continuous axis

from keras.utils.np_utils import to_categorical
print(y_train_label_id.shape)

y_cat_train_label_id=to_categorical(y_train_label_id)
y_cat_test_label_id=to_categorical(y_test_label_id)
# def build_model():
#     mobilenetv2 = MobileNetV2(input_shape=(32, 32, 3), alpha=1, weights=None, include_top=False,classes=120)

    #Varianta 1
#     model=Sequential(mobilenetv2.layers)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dense(120, activation='softmax'))
    
    # Varianta 2
#     model = Sequential([mobilenetv2,Dense(120, activation='softmax')])
    
    
    # Varianta 3
#     model=Sequential()
#     model.add(mobilenetv2)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dense(120, activation='softmax'))
    
   
#     return model

model = MobileNetV2(input_shape=(32, 32, 3), alpha=1, weights=None,classes=131)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())


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
model.load_weights("weights-improvement-12-0.95.hdf5")

# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
print("Created model and loaded weights from file")
model.evaluate(X_test,y_cat_test_label_id)


#model.save("MobileNetV2 - 131 - 2.0.h5")