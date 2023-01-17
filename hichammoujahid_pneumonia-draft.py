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
from keras.applications import VGG16, VGG19
dataset_name='chest-xray'
dataset_path = os.path.join('../input/', dataset_name)

nbr_batch_size=16 

epochs = 25

classes=os.listdir(dataset_path)
print(classes)

num_classes = len(classes)

labels, data= [], []
for class_name in tqdm(sorted(os.listdir(dataset_path))):
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
print(f'Nombre of Normal : {(labelsV2 == 0).sum()}')
print(f'Nombre of Pneumonia : {(labelsV2 == 1).sum()}')

data_Train, data_Test, labels_Train, labels_Test = train_test_split(dataV2, labelsV2, test_size=0.3 , random_state=0, stratify=labels) #, 
data_Test, data_Val, labels_Test, labels_Val = train_test_split(data_Test, labels_Test, test_size=0.5 , random_state=0, stratify=labels_Test)

labels_Train_ctg = np_utils.to_categorical(labels_Train, num_classes)
labels_Val_ctg = np_utils.to_categorical(labels_Val, num_classes)
labels_Test_ctg = np_utils.to_categorical(labels_Test, num_classes)

print("Labels_Train")
print(f'Nombre of Normal : {(labels_Train == 0).sum()}')
print(f'Nombre of Pneumonia : {(labels_Train == 1).sum()}')

print("Labels_Val")
print(f'Nombre of Normal : {(labels_Val == 0).sum()}')
print(f'Nombre of Pneumonia : {(labels_Val == 1).sum()}')

print("Labels_Test")
print(f'Nombre of Normal : {(labels_Test == 0).sum()}')
print(f'Nombre of Pneumonia : {(labels_Test == 1).sum()}')
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
                               ) #


plt.figure(figsize=(16, 6))

plt.suptitle('VGG 16 Performance Model ', fontsize=14)

ax = plt.subplot(121)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg16.history['acc'], label='Training')
plt.plot(history_vgg16.history['val_acc'], label='Validation')
plt.title('Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()


ax = plt.subplot(122)
plt.plot([0, 0.4], [0, 0.4], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg16.history['loss'], label='Training')
plt.plot(history_vgg16.history['val_loss'], label='Validation')
plt.title('Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()

plt.savefig('VGG 16 Performance Model .png')
plt.show()
predict_labels_Test = model_vgg16.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Nombre of Normal : {(predict_labels_Ar == 0).sum()}')
print(f'Nombre of Pneumonia : {(predict_labels_Ar == 1).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test) 

plt.figure()
ax= plt.subplot()
sns.set(font_scale=1)
sns.heatmap(cm, annot= True, fmt='', cmap='GnBu', cbar=True)
labels=["Normal","Pneumonia"]

ax.set_xlabel("\nTrue Labels\n")
ax.set_ylabel("Predicted Labels\n")
ax.set_title('Confusion Matrix of VGG 16 Model'); 
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels);
plt.savefig('Confusion Matrix of VGG 16 Model.png')
score_vgg16 = model_vgg16.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_vgg16[0]) 
print('Test accuracy:', score_vgg16[1])

auc = roc_auc_score(labels_Test, predict_labels)
auc_vgg16 = 'VGG16_AUC = {}'.format("%.2f" % auc)

f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
ax.set_facecolor('white')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = "1.25"
plt.title('ROC Curve of VGG 16 Model', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr_vgg16, tpr_vgg16, thresholds_vgg16 = roc_curve(labels_Test, model_vgg16.predict_proba(data_Test)[:,1]) 
sns.lineplot(x=fpr_vgg16, y=tpr_vgg16, marker='.' ,color=sns.color_palette("husl", 8)[-2], linewidth=2, label=auc_vgg16)


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

plt.figure(figsize=(16, 6))

plt.suptitle('VGG 19 Performance Model ', fontsize=12)

ax = plt.subplot(121)
plt.plot([0.9, 1], [0.9, 1], ' ',color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg19.history['acc'], label='Training')
plt.plot(history_vgg19.history['val_acc'], label='Validation')
plt.title('Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()



ax = plt.subplot(122)
plt.plot([0, 0.4], [0, 0.4], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_vgg19.history['loss'], label='Training')
plt.plot(history_vgg19.history['val_loss'], label='Validation')
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('VGG 19 Performance Model .png')
plt.show()

predict_labels_Test = model_vgg19.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Number of Normal : {(predict_labels_Ar == 0).sum()}')
print(f'Number of Pneumonia : {(predict_labels_Ar == 1).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test) 

plt.figure()
ax= plt.subplot()
sns.set(font_scale=1)
sns.heatmap(cm, annot= True, fmt='', cmap='GnBu', cbar=True)
labels=["Normal", "Pneumonia"]

ax.set_xlabel("\nTrue Labels\n")
ax.set_ylabel("Predicted Labels\n")
ax.set_title('Confusion Matrix of VGG 19 Model'); 
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels);
plt.savefig('Confusion Matrix of VGG 19 Model.png')

score_vgg19 = model_vgg19.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_vgg19[0]) 
print('Test accuracy:', score_vgg19[1])

auc = roc_auc_score(labels_Test, predict_labels)
auc_vgg19 = 'VGG19_AUC = {}'.format("%.2f" % auc)

f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
ax.set_facecolor('white')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = "1.25"
plt.title('ROC Curve of VGG 19 Model', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr_vgg19, tpr_vgg19, thresholds_vgg19 = roc_curve(labels_Test, model_vgg19.predict_proba(data_Test)[:,1]) 
sns.lineplot(x=fpr_vgg19, y=tpr_vgg19, marker='.' ,color=sns.color_palette("husl", 8)[-2], linewidth=2, label=auc_vgg19)
conv_base = NASNetMobile(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

conv_base.trainable = True

model_nASNetMobile = Sequential()
model_nASNetMobile.add(conv_base)
model_nASNetMobile.add(layers.Flatten())
model_nASNetMobile.add(layers.Dense(512, activation='relu'))
model_nASNetMobile.add(layers.Dropout(0.25))
model_nASNetMobile.add(layers.Dense(num_classes, activation='softmax'))

model_nASNetMobile.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-6), metrics=['acc'])

history_nASNetMobile = model_nASNetMobile.fit(data_Train,labels_Train_ctg
                                ,batch_size=64, epochs=25
                                ,validation_data=(data_Val,labels_Val_ctg)
                                             )
plt.figure(figsize=(16, 6))

plt.suptitle('NASNetMobile Performance Model ', fontsize=12)

ax = plt.subplot(121)
plt.plot([0.9, 1], [0.9, 1], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_nASNetMobile.history['acc'], label='Training')
plt.plot(history_nASNetMobile.history['val_acc'], label='Validation')
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


ax =plt.subplot(122)
plt.plot([0, 0.4], [0, 0.4], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_nASNetMobile.history['loss'], label='Training')
plt.plot(history_nASNetMobile.history['val_loss'], label='Validation')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('NASNetMobile Performance Model .png')
plt.show()

predict_labels_Test = model_nASNetMobile.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Number of normal : {(predict_labels_Ar == 0).sum()}')
print(f'Number of pneumonia : {(predict_labels_Ar == 1).sum()}')


print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test) 

plt.figure()
ax= plt.subplot()
sns.set(font_scale=1)
sns.heatmap(cm, annot= True, fmt='', cmap='GnBu', cbar=True)
labels=["Normal","Pneumonia"]

ax.set_xlabel("\nTrue Labels\n")
ax.set_ylabel("Predicted Labels\n")
ax.set_title('Confusion Matrix of NASNetMobile Model'); 
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels);
plt.savefig('Confusion Matrix of NASNetMobile Model.png')

score_nASNetMobile = model_nASNetMobile.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_nASNetMobile[0]) 
print('Test accuracy:', score_nASNetMobile[1])

auc = roc_auc_score(labels_Test, predict_labels)
auc_nASNetMobile = 'NASNetMobile_AUC = {}'.format("%.2f" % auc)

f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
ax.set_facecolor('white')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = "1.25"
plt.title('ROC Curve of NASNetMobile Model', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr_nASNetMobile, tpr_nASNetMobile, thresholds_nASNetMobile = roc_curve(labels_Test, model_nASNetMobile.predict_proba(data_Test)[:,1]) 
sns.lineplot(x=fpr_nASNetMobile, y=tpr_nASNetMobile, marker='.' ,color=sns.color_palette("husl", 8)[-2], linewidth=2, label=auc_nASNetMobile)
conv_base = ResNet152V2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

conv_base.trainable = True

model_resNet152V2 = Sequential()
model_resNet152V2.add(conv_base)
model_resNet152V2.add(layers.Flatten())
model_resNet152V2.add(layers.Dense(1024, activation='relu'))
model_resNet152V2.add(layers.Dropout(0.25))
model_resNet152V2.add(layers.Dense(512, activation='relu'))
model_resNet152V2.add(layers.Dense(num_classes, activation='softmax'))

model_resNet152V2.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-6), metrics=['acc'])

history_resNet152V2 = model_resNet152V2.fit(data_Train,labels_Train_ctg
                                ,batch_size=32, epochs=200
                                ,validation_data=(data_Val,labels_Val_ctg)
                                ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')]                               
                               )
plt.figure(figsize=(16, 6))

plt.suptitle('ResNet152V2 Performance Model ', fontsize=12)

ax = plt.subplot(121)
plt.plot([0.9, 1], [0.9, 1], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_resNet152V2.history['acc'], label='Training')
plt.plot(history_resNet152V2.history['val_acc'], label='Validation')
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


ax = plt.subplot(122)
plt.plot([0, 0.4], [0, 0.4], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_resNet152V2.history['loss'], label='Training')
plt.plot(history_resNet152V2.history['val_loss'], label='Validation')
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('ResNet152V2 Performance Model .png')
plt.show()


predict_labels_Test = model_resNet152V2.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Number of Normal : {(predict_labels_Ar == 0).sum()}')
print(f'Number of Pneumonia : {(predict_labels_Ar == 1).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test) 

plt.figure()
ax= plt.subplot()
sns.set(font_scale=1)
sns.heatmap(cm, annot= True, fmt='', cmap='GnBu', cbar=True)
labels=["Normal","Pneumonia"]

ax.set_xlabel("\nTrue Labels\n")
ax.set_ylabel("Predicted Labels\n")
ax.set_title('Confusion Matrix of ResNet152V2 Model'); 
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels);
plt.savefig('Confusion Matrix of ResNet152V2 Model.png')

score_resNet152V2 = model_resNet152V2.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_resNet152V2[0]) 
print('Test accuracy:', score_resNet152V2[1])

auc = roc_auc_score(labels_Test, predict_labels)
auc_resNet152V2 = 'ResNet152V2_AUC = {}'.format("%.2f" % auc)

f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
ax.set_facecolor('white')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = "1.25"
plt.title('ROC Curve of ResNet152V2 Model', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr_resNet152V2, tpr_resNet152V2, thresholds_resNet152V2 = roc_curve(labels_Test, model_resNet152V2.predict_proba(data_Test)[:,1]) 
sns.lineplot(x=fpr_resNet152V2, y=tpr_resNet152V2, marker='.' ,color=sns.color_palette("husl", 8)[-2], linewidth=2, label=auc_resNet152V2)

conv_base = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

conv_base.trainable = True

model_inceptionResNetV2 = Sequential()
model_inceptionResNetV2.add(conv_base)
model_inceptionResNetV2.add(layers.Flatten())
model_inceptionResNetV2.add(layers.Dense(1024, activation='relu'))
model_inceptionResNetV2.add(layers.Dropout(0.25))
model_inceptionResNetV2.add(layers.Dense(512, activation='relu'))
model_inceptionResNetV2.add(layers.Dense(num_classes, activation='sigmoid'))

model_inceptionResNetV2.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-6), metrics=['acc'])

model_inceptionResNetV2.summary()

history_inceptionResNetV2 = model_inceptionResNetV2.fit(data_Train,labels_Train_ctg
                                ,batch_size=64, epochs=100
                                ,validation_data=(data_Val,labels_Val_ctg)
                                ,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')]                               
                               )
plt.figure(figsize=(16, 6))

plt.suptitle('InceptionResNetV2 Performance Model ', fontsize=12)

ax =plt.subplot(121)
plt.plot([0.9, 1], [0.9, 1], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_inceptionResNetV2.history['acc'], label='Training')
plt.plot(history_inceptionResNetV2.history['val_acc'], label='Validation')
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


ax = plt.subplot(122)
plt.plot([0, 0.4], [0, 0.4], ' ', color='silver')
ax.set_facecolor('white')
plt.plot(history_inceptionResNetV2.history['loss'], label='Training')
plt.plot(history_inceptionResNetV2.history['val_loss'], label='Validation')
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('InceptionResNetV2 Performance Model .png')
plt.show()

predict_labels_Test = model_inceptionResNetV2.predict(data_Test)

predict_labels=np.argmax(predict_labels_Test, axis=1)
# print(predict_labels)

predict_labels_TestV2_ctg = np_utils.to_categorical(predict_labels, num_classes)

predict_labels_Ar = np.asarray(predict_labels)
print("\npredict_labels_Test")
print(f'Number of Normal : {(predict_labels_Ar == 0).sum()}')
print(f'Number of Pneumonia : {(predict_labels_Ar == 1).sum()}')

print("\n"+classification_report(predict_labels_TestV2_ctg, labels_Test_ctg))

cm = confusion_matrix(predict_labels, labels_Test) 

plt.figure()
ax= plt.subplot()
sns.set(font_scale=1)
sns.heatmap(cm, annot= True, fmt='', cmap='GnBu', cbar=True)
labels=["Normal","Pneumonia"]

ax.set_xlabel("\nTrue Labels\n")
ax.set_ylabel("Predicted Labels\n")
ax.set_title('Confusion Matrix of InceptionResNetV2 Model'); 
ax.xaxis.set_ticklabels(labels); 
ax.yaxis.set_ticklabels(labels);
plt.savefig('Confusion Matrix of InceptionResNetV2 Model.png')


score_inceptionResNetV2 = model_inceptionResNetV2.evaluate(data_Test,labels_Test_ctg, verbose = 0)

print('Test loss:', score_inceptionResNetV2[0]) 
print('Test accuracy:', score_inceptionResNetV2[1])

auc = roc_auc_score(labels_Test, predict_labels)
auc_inceptionResNetV2 = 'InceptionResNetV2_AUC = {}'.format("%.2f" % auc)

f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
ax.set_facecolor('white')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = "1.25"
plt.title('ROC Curve of InceptionResNetV2 Model', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr_inceptionResNetV2, tpr_inceptionResNetV2, thresholds_inceptionResNetV2 = roc_curve(labels_Test, model_inceptionResNetV2.predict_proba(data_Test)[:,1]) 
sns.lineplot(x=fpr_inceptionResNetV2, y=tpr_inceptionResNetV2, marker='.' ,color=sns.color_palette("husl", 8)[-2], linewidth=2, label=auc_inceptionResNetV2)



f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
ax.set_facecolor('white')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = "1.25"
sns.lineplot(x=fpr_vgg16, y=tpr_vgg16, linewidth=2, label=auc_vgg16)
sns.lineplot(x=fpr_vgg19, y=tpr_vgg19, linewidth=2, label=auc_vgg19)
sns.lineplot(x=fpr_nASNetMobile, y=tpr_nASNetMobile, linewidth=2, label=auc_nASNetMobile)
sns.lineplot(x=fpr_resNet152V2, y=tpr_resNet152V2, linewidth=2, label=auc_resNet152V2)
sns.lineplot(x=fpr_inceptionResNetV2, y=tpr_inceptionResNetV2, linewidth=2, label=auc_inceptionResNetV2)
plt.title('ROC Curves performance evaluation models', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend()