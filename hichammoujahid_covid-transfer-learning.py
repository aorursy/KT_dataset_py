#https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import cv2
from tqdm import tqdm
%matplotlib inline
 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras import layers
from keras.applications import *
from keras.preprocessing.image import load_img
import random
#from tensorflow.keras.applications import EfficientNetB7
#1. Creating and compiling the model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import tensorflow as tf
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import tensorflow as tf
import time
from keras.preprocessing import image
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import *
import itertools
import matplotlib.pyplot as plt
import os
import cv2
import imghdr
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import VGG16, VGG19, MobileNetV2
dataset_name='covid19-dataset'
dataset_path = os.path.join('../input/', dataset_name)

nbr_batch_size=16 

epochs = 25

classes=sorted(os.listdir(dataset_path))
print(classes)

num_classes = len(classes)

labels, data= [], []
for class_name in tqdm(classes):
    class_path = os.path.join(dataset_path, class_name) 
    class_id = classes.index(class_name)
    for path in os.listdir(class_path):
        path = os.path.join(class_path, path)
        if imghdr.what(path) == None:
            # this is not an image file
            continue
        image = cv2.imread(path)
        image= cv2.resize(image, (224,224))
        data.append(image)
        labels.append(class_id) #class_id
dataV2 = np.array(data)
labelsV2 = np.asarray(labels)

print("Dataset")
print(f'Nombre of {classes[0]} : {(labelsV2 == 0).sum()}')
print(f'Nombre of {classes[1]} : {(labelsV2 == 1).sum()}')
print(f'Nombre of {classes[2]} : {(labelsV2 == 2).sum()}')

data_Train, data_Test, labels_Train, labels_Test = train_test_split(dataV2, labelsV2, test_size=0.3 , random_state=0, stratify=labels) #, 
data_Test, data_Val, labels_Test, labels_Val = train_test_split(data_Test, labels_Test, test_size=0.5 , random_state=0, stratify=labels_Test)

labels_Train_ctg = np_utils.to_categorical(labels_Train, num_classes)
labels_Val_ctg = np_utils.to_categorical(labels_Val, num_classes)
labels_Test_ctg = np_utils.to_categorical(labels_Test, num_classes)

print("Labels_Train")
print(f'Nombre de {classes[0]} : {(labels_Train == 0).sum()}')
print(f'Nombre de {classes[1]} : {(labels_Train == 1).sum()}')
print(f'Nombre de {classes[2]} : {(labels_Train == 2).sum()}')

print("Labels_Val")
print(f'Nombre de {classes[0]} : {(labels_Val == 0).sum()}')
print(f'Nombre de {classes[1]} : {(labels_Val == 1).sum()}')
print(f'Nombre de {classes[2]} : {(labels_Val == 2).sum()}')
print("Labels_Test")
print(f'Nombre de {classes[0]} : {(labels_Test == 0).sum()}')
print(f'Nombre de {classes[1]} : {(labels_Test == 1).sum()}')
print(f'Nombre de {classes[2]} : {(labels_Test == 2).sum()}')
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

conv_base.trainable = True

model_vgg16 = Sequential()
model_vgg16.add(conv_base)
model_vgg16.add(layers.Flatten())
model_vgg16.add(layers.Dense(512, activation='relu'))
model_vgg16.add(layers.Dropout(0.25))
model_vgg16.add(layers.Dense(256, activation='relu'))
model_vgg16.add(layers.Dense(num_classes, activation='sigmoid'))

model_vgg16.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-6), metrics=['acc'])
model_vgg16.summary()

history_vgg16 = model_vgg16.fit(data_Train,labels_Train_ctg
                                ,batch_size=64, epochs=200
                                ,validation_data=(data_Val,labels_Val_ctg)
                                ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')]                               
                               )#
import matplotlib

andy_theme = {'axes.grid': True,
              'grid.linestyle': '--',
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 14,
              'legend.title_fontsize': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.labelsize': 16,
              'axes.titlesize': 20,
              'axes.linewidth':'1',
              'axes.edgecolor':'0',
              'figure.dpi': 600
               }
print(f"VGG-16 Accuracy :\n {history_vgg16.history['acc']}\n")
print(f"VGG-16 LOSS :\n {history_vgg16.history['val_acc']}\n")
vgg161=[0.43030795454978943, 0.5782009959220886, 0.6349270939826965, 0.7066450715065002, 0.756077766418457, 0.8136142492294312, 0.8504862189292908, 0.8772284984588623, 0.9076175093650818, 0.9161264300346375, 0.9327390789985657, 0.9388168454170227, 0.9469205737113953, 0.9493517279624939, 0.9501620531082153, 0.9635332226753235, 0.9647487998008728, 0.9647487998008728, 0.9696110486984253, 0.974473237991333, 0.9756888151168823, 0.9801458716392517, 0.9760940074920654, 0.9817666411399841]
vgg162=[0.5160680413246155, 0.6162570714950562, 0.6937618255615234, 0.7410207986831665, 0.7655954360961914, 0.8525519967079163, 0.8733459115028381, 0.9092627763748169, 0.9281663298606873, 0.9281663298606873, 0.9319470524787903, 0.9489603042602539, 0.9489603042602539, 0.9527410268783569, 0.9489603042602539, 0.9508506655693054, 0.9546313881874084, 0.9546313881874084, 0.9546313881874084, 0.95652174949646, 0.9584121108055115, 0.9584121108055115, 0.9546313881874084, 0.9584121108055115]
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='in', length=5, width=1, colors='black',
               grid_color='black', grid_alpha=0.5)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg16.history['acc'], label='Training', linewidth=3)
plt.plot(history_vgg16.history['val_acc'], label='Validation', linewidth=3)
plt.xlabel('Epochs', fontsize=14, fontweight='bold' )
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.rcParams.update(andy_theme)
plt.legend(loc='lower right')
plt.rc('text', usetex=False)
plt.rc('axes', linewidth=1.5)
plt.show()
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00, length=5)
ax.tick_params(direction='in', length=5, width=1, colors='black',
               grid_color='black', grid_alpha=0.5)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg16.history['loss'], label='Training', linewidth=3)
plt.plot(history_vgg16.history['val_loss'], label='Validation', linewidth=3)
plt.xlabel('Epochs', fontsize=14, fontweight='bold' , color="black")
plt.ylabel('Loss', fontsize=14, fontweight='bold', color="black")
plt.rcParams.update(andy_theme)
plt.legend(loc='upper right')
plt.rc('text', usetex=False)
plt.rc('axes', linewidth=1.5)
#plt.rc('font', weight='bold')
plt.show()
predict_labels_Test = model_vgg16.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Nombre of {classes[0]} : {(predict_labels_Ar == 0).sum()}')
print(f'Nombre of {classes[1]} : {(predict_labels_Ar == 1).sum()}')
print(f'Nombre of {classes[2]} : {(predict_labels_Ar == 2).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test) 
cm1=np.array([[126, 1, 3], [0,193,9],[0,7,190]])

plt.figure()
ax= plt.subplot()
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='out', length=5, width=2, colors='black',
               grid_color='black', grid_alpha=0.5)
sns.set(font_scale=2)
sns.heatmap(cm1, annot= True, fmt='', cmap='GnBu', cbar=True, annot_kws={"size": 16})
labels=classes
plt.rcParams.update(andy_theme)
ax.set_xlabel("\nTrue Labels\n" ,fontweight="bold")
ax.set_ylabel("Predicted Labels\n" ,fontweight="bold")
#ax.set_title('Confusion Matrix of VGG 16 Model',fontweight="bold"); 
ax.xaxis.set_ticklabels(labels,fontweight="bold"); 
ax.yaxis.set_ticklabels(labels,fontweight="bold");

plt.show()
score_vgg16 = model_vgg16.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_vgg16[0]) 
print('Test accuracy:', score_vgg16[1])
conv_base = VGG19(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

conv_base.trainable = True

model_vgg19 = Sequential()
model_vgg19.add(conv_base)
model_vgg19.add(layers.Flatten())
model_vgg19.add(layers.Dense(512, activation='relu'))
model_vgg19.add(layers.Dropout(0.25))
model_vgg19.add(layers.Dense(256, activation='relu'))
model_vgg19.add(layers.Dense(num_classes, activation='sigmoid'))

model_vgg19.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-6), metrics=['acc'])

history_vgg19 = model_vgg19.fit(data_Train,labels_Train_ctg
                                ,batch_size=64, epochs=200
                                ,validation_data=(data_Val,labels_Val_ctg)
                                ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')]                               
                               ) #
print(f"VGG-19 Accuracy :\n {history_vgg19.history['acc']}\n")
print(f"VGG-19 LOSS :\n {history_vgg19.history['val_acc']}\n")
#Accuracy
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='in', length=5, width=1, colors='black',
               grid_color='black', grid_alpha=0.5)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg19.history['acc'], label='Training', linewidth=3)
plt.plot(history_vgg19.history['val_acc'], label='Validation', linewidth=3)
plt.xlabel('Epochs', fontsize=14, fontweight='bold' , color="black")
plt.ylabel('Accuracy', fontsize=14, fontweight='bold', color="black")
plt.rcParams.update(andy_theme)
plt.legend(loc='lower right')
plt.rc('text', usetex=False)
plt.rc('axes', linewidth=1.5)
#plt.rc('font', weight='bold')
plt.show()
#Loss
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(direction='in', length=5, width=1, colors='black',
               grid_color='black', grid_alpha=0.5)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg19.history['loss'], label='Training', linewidth=3)
plt.plot(history_vgg19.history['val_loss'], label='Validation', linewidth=3)
plt.xlabel('Epochs', fontsize=14, fontweight='bold' , color="black" )
plt.ylabel('Loss', fontsize=14, fontweight='bold', color="black")
plt.rcParams.update(andy_theme)
plt.legend(loc='upper right')
plt.rc('text', usetex=False)
plt.rc('axes', linewidth=1.5)
#plt.rc('font', weight='bold')
plt.show()

#Confusion matrix

predict_labels_Test = model_vgg19.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Number of Covid-19 : {(predict_labels_Ar == 0).sum()}')
print(f'Number of Normal : {(predict_labels_Ar == 1).sum()}')
print(f'Number of Pneumonia : {(predict_labels_Ar == 2).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test)
cm2=np.array([[126, 0, 1], [0,194,8],[0,7,193]])

plt.figure()
ax= plt.subplot()
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='out', length=5, width=2, colors='black',
               grid_color='black', grid_alpha=0.5)
sns.set(font_scale=2)
sns.heatmap(cm2, annot= True, fmt='', cmap='GnBu', cbar=True, annot_kws={"size": 16})
labels=["Covid-19", "Normal","Pneumonia"]
plt.rcParams.update(andy_theme)
ax.set_xlabel("\nTrue Labels\n" ,fontweight="bold")
ax.set_ylabel("Predicted Labels\n" ,fontweight="bold")
#ax.set_title('Confusion Matrix of VGG 16 Model',fontweight="bold"); 
ax.xaxis.set_ticklabels(labels,fontweight="bold"); 
ax.yaxis.set_ticklabels(labels,fontweight="bold");

plt.show()


score_vgg19 = model_vgg19.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_vgg19[0]) 
print('Test accuracy:', score_vgg19[1])
conv_base = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

conv_base.trainable = True

model_mobilenet = Sequential()
model_mobilenet.add(conv_base)
model_mobilenet.add(layers.Flatten())
model_mobilenet.add(layers.Dense(512, activation='relu'))
model_mobilenet.add(layers.Dropout(0.25))
model_mobilenet.add(layers.Dense(256, activation='relu'))
model_mobilenet.add(layers.Dense(num_classes, activation='sigmoid'))

model_mobilenet.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-6), metrics=['acc'])

history_mobilenet = model_mobilenet.fit(data_Train,labels_Train_ctg
                                ,batch_size=64, epochs=200
                                ,validation_data=(data_Val,labels_Val_ctg)
                                ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')]                               
                               ) #
print(f"MobileNet Accuracy :\n {history_mobilenet.history['acc']}\n")
print(f"MobileNet LOSS :\n {history_mobilenet.history['val_acc']}\n")
#Accuracy
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='in', length=5, width=1, colors='black',
               grid_color='black', grid_alpha=0.5)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_mobilenet.history['acc'], label='Training', linewidth=3)
plt.plot(history_mobilenet.history['val_acc'], label='Validation', linewidth=3)
plt.xlabel('Epochs', fontsize=14, fontweight='bold' , color="black")
plt.ylabel('Accuracy', fontsize=14, fontweight='bold', color="black")
plt.rcParams.update(andy_theme)
plt.legend(loc='lower right')
plt.rc('text', usetex=False)
plt.rc('axes', linewidth=1.5)
#plt.rc('font', weight='bold')
plt.show()
#Loss
plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='in', length=5, width=1, colors='black',
               grid_color='black', grid_alpha=0.5)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_mobilenet.history['loss'], label='Training', linewidth=3)
plt.plot(history_mobilenet.history['val_loss'], label='Validation', linewidth=3)
plt.xlabel('Epochs', fontsize=14, fontweight='bold', color="black" )
plt.ylabel('Loss', fontsize=14, fontweight='bold', color="black")
plt.rcParams.update(andy_theme)
plt.legend(loc='upper right')
plt.rc('text', usetex=False)
plt.rc('axes', linewidth=1.5)
#plt.rc('font', weight='bold')
plt.show()
predict_labels_Test = model_mobilenet.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Number of Covid-19 : {(predict_labels_Ar == 0).sum()}')
print(f'Number of Normal : {(predict_labels_Ar == 1).sum()}')
print(f'Number of Pneumonia : {(predict_labels_Ar == 2).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test)
cm3=np.array([[120, 0, 0], [3,197,12],[3,4,190]])

plt.figure()
ax= plt.subplot()
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='major', width=1.00)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=2.5)
ax.tick_params(direction='out', length=5, width=2, colors='black',
               grid_color='black', grid_alpha=0.5)
sns.set(font_scale=2)
sns.heatmap(cm3, annot= True, fmt='', cmap='GnBu', cbar=True, annot_kws={"size": 16})
labels=["Covid-19", "Normal","Pneumonia"]
plt.rcParams.update(andy_theme)
ax.set_xlabel("\nTrue Labels\n" ,fontweight="bold")
ax.set_ylabel("Predicted Labels\n" ,fontweight="bold")
ax.xaxis.set_ticklabels(labels,fontweight="bold"); 
ax.yaxis.set_ticklabels(labels,fontweight="bold");
plt.show()
score_mobilenet = model_mobilenet.evaluate(data_Test,labels_Test_ctg, verbose = 0)
print('Test loss:', score_mobilenet[0]) 
print('Test accuracy:', score_mobilenet[1])