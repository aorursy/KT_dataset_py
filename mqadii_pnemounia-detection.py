# General libraries
import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools    
from keras.optimizers import SGD , RMSprop



%matplotlib inline


# Deep learning libraries
import tensorflow as tf

import tensorflow.keras.backend as K
from  keras.models import Model, Sequential
from  keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from  keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from  keras.optimizers import Adam
from  keras.preprocessing.image import ImageDataGenerator
from  keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.backend import manual_variable_initialization 



data_gen = ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=5,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.15,
                             );
image_size=(224,224)
labels=['PNEUMONIA','NORMAL']
train_path="/kaggle/input/chest-xray-custom/chest_xray/chest_xray/train"
test_path="/kaggle/input/chest-xray-custom/chest_xray/chest_xray/test"
# valid_path="/kaggle/input/pnemounia-validation-images/validation images"
valid_path="/kaggle/input/chest-xray-custom/chest_xray/chest_xray/val"
plot_images=data_gen.flow_from_directory(valid_path,target_size=image_size,class_mode='binary',batch_size=10,color_mode="rgb")

train_gen=data_gen.flow_from_directory(train_path,
                                       target_size=image_size,
                                       class_mode='binary',
                                       batch_size=32,
                                       color_mode="grayscale",
                                      shuffle=True,
                                    seed=15
                                      )
test_gen=data_gen.flow_from_directory(test_path,target_size=image_size,class_mode='binary',batch_size=16,color_mode="grayscale",
                                      shuffle=True,
    seed=15)
valid_gen=data_gen.flow_from_directory(valid_path,target_size=image_size,class_mode='binary',batch_size=1,color_mode="grayscale",
                                     )

x,y=next(plot_images)
print(train_gen.class_indices)
print(test_gen.class_indices)
print(valid_gen.class_indices)


w=100
h=100
fig=plt.figure(figsize=(20, 20))
columns = 10
rows = 1
for i in range(1, 10):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(x[i],cmap='gray')
    plt.text(x=50, y=-10, s=y[i], fontsize=12)
plt.show()
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu",input_shape=(224,224,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu"))

model.add(MaxPool2D((2,2) , strides = (2,2) , padding = 'same'))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D((2,2) , strides = (2,2) , padding = 'same'))

model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D((2,2) , strides = (2,2) , padding = 'same'))

model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D((2,2) , strides = (2,2) , padding = 'same'))


model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.45))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.2))




model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint('best.h5', verbose=1, 
                                save_best_only=True,
                                monitor='val_acc',
                                mode='auto')


   
    
manual_variable_initialization(True)
history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=128,
                    epochs=20,
                    validation_steps=39,
                    validation_data=test_gen,
#                     callbacks = [checkpoint]
                   )
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict_generator(valid_gen,steps=616,verbose=0)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plt.figure()

true = valid_gen.classes[valid_gen.index_array]
pred= predictions.round()
cm = confusion_matrix(true, pred)


plot_confusion_matrix(cm, classes=['Normal','pnemonia'], normalize=False,
                      title='confusion matrix')
plt.show()




from sklearn.metrics import classification_report

print(classification_report(true, pred,target_names=['Normal','pnemonia']))

model.save('final-final.h5')