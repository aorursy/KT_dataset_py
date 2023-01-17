from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
def graph(history):
    epoklar = range(1,len(history["loss"])+1)
    
    plt.plot(epoklar,history["loss"],label="Training Loss")
    plt.plot(epoklar,history["val_loss"],label="Validation Loss")
    plt.title("Loss")
    plt.legend()
    plt.show()
    
    plt.plot(epoklar,history["acc"],label="Training Accuracy")
    plt.plot(epoklar,history["val_acc"],label="Validation Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.show()
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
def resize(images):
    yeni = np.zeros((images.shape[0],96,96,3),dtype=np.float32)
    for i in range(len(images)):
        yeni[i] = cv2.resize(images[i,:,:,:],(96,96))
    return yeni
train_images = resize(train_images)
test_images = resize(test_images)
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
numberOfClass = len(np.unique(train_labels))
numberOfClass
plt.imshow(test_images[20].astype(np.uint8))
plt.axis("off")
plt.show()
plt.imshow(train_images[20].astype(np.uint8))
plt.axis("off")
plt.show()
#one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#x_train,x_valid,y_train,y_valid = train_test_split(train_images,train_labels,test_size = 0.3, random_state = 40,stratify = train_labels)

x_train = train_images[:42500]
y_train = train_labels[:42500]
x_valid = train_images[42500:]
y_valid = train_labels[42500:]

x_test = test_images
y_test = test_labels
#data augmentation and normalize images with using ImageDataGenerator
train_datagen = image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.15,
      height_shift_range=0.15,
      shear_range=0.15,
      zoom_range=0.1,
      horizontal_flip=True,
      fill_mode='nearest')

valid_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255)
def generator(data,x,y,batch_size):
    return data.flow(x,y,batch_size=batch_size)    
batch_size = 128
epoch = 30
size = x_train.shape[1:]
train_generator = generator(train_datagen,x_train,y_train,batch_size=batch_size)
valid_generator = generator(valid_datagen,x_valid,y_valid,batch_size=batch_size)
test_generator =  generator(test_datagen,x_test,y_test,batch_size=batch_size)
vgg16 = VGG16(weights="imagenet",include_top=False,input_shape=size)
vgg16.summary()
def model(train, valid, epochs):
    


    #create new model and add pretrained model in this model
    model = models.Sequential()
    for i in range(len(vgg16.layers)):
        model.add(vgg16.layers[i])

    for i in range(len(model.layers)): #freeze pretrained model
        model.layers[i].trainable = False

    #add new layers your model
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.40))
    model.add(layers.Dense(256,activation="relu"))
    model.add(layers.Dense(numberOfClass,activation="softmax"))

    model.compile(optimizer="adam",loss="categorical_crossentropy",
                  metrics=["acc",tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

    history = model.fit_generator(generator=train,
                    steps_per_epoch=len(train),epochs=epochs,
                    validation_data=valid,
                    validation_steps=len(valid))
    return model, history
Model = model(train=train_generator,valid=valid_generator,epochs=epoch)
graph(Model[1].history)
loss,acc,recall,precision = Model[0].evaluate_generator(test_generator,
                        steps=len(test_generator))
print("Test Accuracy =",acc,'\t',"Test Recall ",recall,'\t',"Test Precision",precision)
from keras import optimizers
def TunedModel(train, valid, epochs):


    #create new model and add pretrained model in this model
    model = models.Sequential()
    for i in range(len(vgg16.layers)):
        model.add(vgg16.layers[i])

    #select after blocks block4_conv1 to fine tunnig
    trnable = False
    for i in model.layers:
        if i.name == "block4_conv1":
            trnable = True
        i.trainable = trnable

    #add new layers your model
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.40))
    model.add(layers.Dense(256,activation="relu"))
    model.add(layers.Dense(numberOfClass,activation="softmax"))

    model.compile(optimizer= optimizers.Adam(lr=1e-5),loss="categorical_crossentropy",
                  metrics=["acc",tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

    history = model.fit_generator(generator=train,
                    steps_per_epoch=len(train),epochs=epochs,
                    validation_data=valid,
                    validation_steps=len(valid))
    return model, history
epoch = 50
Model1 = TunedModel(train=train_generator,valid=valid_generator,epochs=epoch)
graph(Model1[1].history)
loss,acc,recall,precision = Model1[0].evaluate_generator(test_generator,
                        steps=len(test_generator))
print("Test Accuracy =",acc,'\t',"Test Recall ",recall,'\t',"Test Precision",precision)
import pickle
import os
model_file = "model.sav"
with open(model_file,mode='wb') as model_f:
    pickle.dump(Model,model_f)
with open(model_file,mode='rb') as model_f:
    model = pickle.load(model_f)
    result = model[0].evaluate_generator(test_generator,
                        steps=len(test_generator))
    print("result:",result)
