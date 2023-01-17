import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
datagen=ImageDataGenerator(rescale=1/255)
train=datagen.flow_from_directory('/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Train',class_mode='binary')
test=datagen.flow_from_directory('/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Test',class_mode='binary')
train.class_indices
input_shape=(256,256,3)
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import RMSprop

model=Sequential()

model.add(Conv2D(64,(2,2),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(Conv2D(256,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(512,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(Dropout(0.3))


model.add(Dropout(0.4))


model.add(Flatten())


model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))


model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.summary()
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

early=EarlyStopping(monitor='accuracy',patience=3,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1,cooldown=0, mode='auto',min_delta=0.0001, min_lr=0)
model.fit(train,epochs=20,validation_data=(test),shuffle=True,callbacks=[early,reduce_lr])
model.save('/kaggle/working/mask_detection_model.h5')
import pandas as pd
loss=pd.DataFrame(model.history.history)
loss.plot()