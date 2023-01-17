import pandas as pd
import cv2
import matplotlib.pyplot as plt
%matplotlib inline 
import os
import pickle
import numpy as np
path="/kaggle/input/diabeticretinopathy-messidor-eyepac-preprocessed/Messidor-2+EyePac_Balanced/"
list_target_folder=os.listdir(path)
list_target_folder
target_data=[]
train_data=[]

for item in list_target_folder:
    image_path=path+"/"+item
    
    images=os.listdir(image_path)
    
    for image in images:
        img=cv2.imread(image_path+"/"+image)
        img=cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
        train_data.append(img)
        target_data.append(int(item))
    
target_data=np.array(target_data)
train_data=np.array(train_data)
train_data.shape
fig, axs = plt.subplots(1,5,figsize=(15, 4),sharey=True)
for i,item in enumerate(range(5)):
    axs[i].imshow(train_data[i])
    axs[i].set_title(target_data[i])
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
train_labels = lb.fit_transform(target_data)
train_labels
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_data,train_labels,test_size=0.3)
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.applications import EfficientNetB7
#import tensorflow.keras.applications.EfficientNetB7
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

resnet_model = ResNet50(input_shape=(512,512,3), weights=None, include_top=False)
resnet_model.load_weights("../input/weight-file/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
for layer in resnet_model.layers:
    layer.trainable = False
for i in range(-5,0):
    resnet_model.layers[i].trainable=True
from tensorflow.keras.layers import BatchNormalization,Dropout
x = Flatten()(resnet_model.output)
x2=BatchNormalization()(x)
x3=Dense(2048, activation='relu')(x2)
x4=Dropout(0.5)(x3)
x5=Dense(2048, activation='relu')(x4)
x6=Dropout(0.5)(x5)
x8=Dense(1024, activation='relu')(x6)
x9=Dropout(0.3)(x8)
x10=Dense(1024,activation="relu")(x9)
x11=Dropout(0.3)(x10)
x12=BatchNormalization()(x11)
prediction = Dense(5, activation='sigmoid')(x12)
resnet_model=Model(inputs=resnet_model.input, outputs=prediction)
resnet_model.summary()
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
lr_schedule =ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = SGD(learning_rate=lr_schedule)
resnet_model.compile(
  optimizer=optimizer,
  loss="categorical_crossentropy",
  metrics=['accuracy']
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      #rotation_range=30,
      shear_range=0.1,
      zoom_range=[0.3,0.5],
      #width_shift_range=0.4,
      #height_shift_range=0.4,
      horizontal_flip=True,
      vertical_flip=True,
  
      fill_mode='nearest')


test_datagen=ImageDataGenerator(rescale=1./225)
train_set=train_datagen.flow(x_train,y_train)
test_set=test_datagen.flow(x_test,y_test)
history=resnet_model.fit_generator(
         train_set,
        validation_data=test_set,
        epochs=2,
        verbose=1
     )

