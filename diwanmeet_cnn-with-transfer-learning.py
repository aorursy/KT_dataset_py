!pip install tensorflow==2.3.0 -q
import tensorflow as tf
import shutil
import random
#import torch
import os
#import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

print(tf.__version__)
#torch.manual_seed(0)
class_names = ['Parasitized','Uninfected']
root_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images'
root_dir2 = '/kaggle/working'
os.mkdir(os.path.join(root_dir2, 'train'))
root_dir3 = '/kaggle/working/train'
source_dirs = ['Parasitized','Uninfected']
if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    #os.mkdir(os.path.join(root_dir2, 'test'))

    for i, d in enumerate(source_dirs):
        shutil.copytree(os.path.join(root_dir,d), os.path.join(root_dir3, class_names[i]))
        #os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))
if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir2, 'test'))

    for c in class_names:
        os.mkdir(os.path.join(root_dir2, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir3, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 1000)
        for image in selected_images:
            source_path = os.path.join(root_dir3, c, image)
            target_path = os.path.join(root_dir2, 'test', c, image)
            shutil.move(source_path, target_path)
dirtrain = "/kaggle/working/train"
training_data = tf.keras.preprocessing.image_dataset_from_directory(directory = dirtrain, subset=None, labels = "inferred", color_mode = "rgb", image_size=(224,224), shuffle = True)
def one_hot_label(image,label):
    label = tf.one_hot(label, 2)
    return image, label

training_data = training_data.map(one_hot_label)
dirtest="/kaggle/working/test"
test_data = tf.keras.preprocessing.image_dataset_from_directory(directory = dirtest, subset=None, labels = "inferred", color_mode = "rgb", image_size=(224,224), shuffle = True)
def one_hot_label(image,label):
    label = tf.one_hot(label, 2)
    return image, label

test_data = test_data.map(one_hot_label)
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
efficientb6 = tf.keras.applications.efficientnet.EfficientNetB6(include_top=None, weights="imagenet", classes=2, input_shape=(224,224,3))
efficientb6.summary()
x= efficientb6.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_EfficientNetB6 = Model(inputs = efficientb6.input, outputs = predictions)
model_EfficientNetB6.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])
historyB6=model_EfficientNetB6.fit(training_data, epochs=2)
efficientb5 = tf.keras.applications.efficientnet.EfficientNetB5(include_top=None, weights="imagenet", classes=2, input_shape=(224,224,3))
x= efficientb5.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_EfficientNetB5 = Model(inputs = efficientb5.input, outputs = predictions)
model_EfficientNetB5.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])
historyB5=model_EfficientNetB5.fit(training_data, epochs=2)
preds= model_EfficientNetB5.evaluate(test_data)
print("Loss: "+str(preds[0]))
print("Accuracy: "+str(preds[1]))
efficientb4 = tf.keras.applications.efficientnet.EfficientNetB4(include_top=None, weights="imagenet", classes=2, input_shape=(224,224,3))
x= efficientb4.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_EfficientNetB4 = Model(inputs = efficientb4.input, outputs = predictions)
model_EfficientNetB4.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
historyb4 = model_EfficientNetB4.fit(training_data, epochs=2)
preds = model_EfficientNetB4.evaluate(test_data)
print("Loss: "+ str(preds[0]))
print("Accuracy: "+ str(preds[1]))
Inceptionresnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=None, weights="imagenet", classes=2, input_shape=(224,224,3))
Inceptionresnet.summary()
x= Inceptionresnet.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_InceptionResnet = Model(inputs = Inceptionresnet.input, outputs = predictions)
model_InceptionResnet.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
historyInceptionResnet = model_InceptionResnet.fit(training_data, epochs=2)
preds = model_InceptionResnet.evaluate(test_data)
print("Loss: "+ str(preds[0]))
print("Accuracy: "+ str(preds[1]))
exception = tf.keras.applications.xception.Xception(include_top=None, weights="imagenet", classes=2, input_shape=(224,224,3))
exception.summary()
x= exception.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_Xception = Model(inputs = exception.input, outputs = predictions)
model_Xception.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
historyXception = model_Xception.fit(training_data, epochs = 2)
preds = model_Xception.evaluate(test_data)
print("Loss: "+ str(preds[0]))
print("Accuracy: "+ str(preds[1]))
model152 = tf.keras.applications.ResNet152V2(include_top=None,input_shape=(224,224,3), classes=2, weights="imagenet")

x= model152.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_resnet152 = Model(inputs = model152.input, outputs = predictions)
model_resnet152.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])
model_resnet152.fit(training_data, epochs=2)
preds = model_resnet152.evaluate(test_data)
print("Loss: "+ str(preds[0]))
print("Accuracy: "+ str(preds[1]))
resnet101 = tf.keras.applications.ResNet101V2(include_top=None,input_shape=(224,224,3), classes=2, weights="imagenet")

x = resnet101.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) 

model_resnet101 = Model(inputs = resnet101.input, outputs = predictions)

model_resnet101.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

model_resnet101.fit(training_data, epochs=2)

preds = model_resnet101.evaluate(test_data)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
vgg19 = tf.keras.applications.vgg19.VGG19(include_top=None, weights="imagenet", input_shape=(224,224,3), classes=2)

x= vgg19.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(28)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Flatten()(x)

predictions = Dense(2, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_vgg19 = Model(inputs = vgg19.input, outputs = predictions)

model_vgg19.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

model_vgg19.fit(training_data, epochs=2)
preds = model_vgg19.evaluate(test_data)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
