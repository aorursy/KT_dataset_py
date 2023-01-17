# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import matplotlib.image as mpimg 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,classification_report
import os 
from glob import glob 
#print(os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val"))
path_train =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
path_test = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test'
path_val= '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'
path_train_pneumonia =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA'
path_train_normal = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL'
path_test_normal = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL'
path_test_pneumonia = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA'
path_val_normal =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL'
path_val_pneumonia =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA'
images_path = glob(path_train_normal+"/*.jpeg")
images = []
for  path in images_path:    
     images.append((path,0))
Pneumonia_images_path =  glob(path_train_pneumonia+"/*.jpeg")
for path in Pneumonia_images_path:
    images.append((path,1))

df = pd.DataFrame(images, columns=['images','labels'])
df.groupby('labels').count().plot.barh()
plt.figure(figsize=(10,10))
im = []
for i in range(0,10):
    im.append(plt.imread(df.loc[i].images))
len(im)
for i in range(0,9):
    plt.subplot(3, 3, i+1)
    plt.imshow(im[i])

IMG_SIZE = 150 
BATCH_SIZE = 32
img_generator =keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                            rotation_range=10,
                                                            shear_range=0.2,
                                                            zoom_range=0.2,
                                                            width_shift_range=0.1,
                                                            height_shift_range=0.1,
                                                            horizontal_flip=True)



train_generator = img_generator.flow_from_directory(path_train,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  #color_mode="grayscale",
                                                  shuffle=True,
                                                  class_mode='binary', 
                                                  subset='training')

val_generator  = img_generator.flow_from_directory(path_val,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  #batch_size= BATCH_SIZE,
                                                  #color_mode="grayscale",
                                                  shuffle=False,
                                                  class_mode='binary',
                                                  )

test_generator = img_generator.flow_from_directory(path_test,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  #color_mode="grayscale",
                                                  shuffle=False,
                                                  class_mode='binary')
plt.imshow(train_generator[0][0][0])
# sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
# layers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001) #0.00001 reduce the learning rate when the metrics isnt improoving
#early_stopping = EarlyStopping(patience = 5)
callback = [learning_rate_reduction]
cnn_model =  Sequential()

cnn_model.add(Conv2D(32,(3,3), activation ='relu', input_shape=(IMG_SIZE , IMG_SIZE,3)))#Convlayer mit 32 Filtern der Größe 3x3. 
cnn_model.add(Conv2D(64, (3,3),activation='relu')) #increase the non-linearity
cnn_model.add(MaxPooling2D(2,2)) #die features-Maps werden als 2x2 array transformiert
#cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(64, (3,3), activation='relu'))
cnn_model.add(Conv2D(64, (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(128, (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))


cnn_model.add(Flatten())#Matrizendaten abflachen, weil die Fully-Connected layer die Daten als Vector nimmt.
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

cnn_model.summary()
model = cnn_model.fit_generator(train_generator,
                                steps_per_epoch = 163,
                                epochs = 10,
                                validation_data=val_generator,
                                validation_steps=len(val_generator),
                                callbacks = callback)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(model.history['val_accuracy'],label='validation Accuracy')
plt.plot(model.history['accuracy'], label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training  Accuracy')

plt.subplot(1, 2, 2)
plt.plot(model.history['val_loss'],label='validation loss')
plt.plot(model.history['loss'], label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training  Loss')

plt.show()
predic= cnn_model.evaluate(test_generator)
predictions =  cnn_model.predict_generator(test_generator)

CNN_prediction_final = np.where(predictions>0.5,1,0)
print(classification_report(test_generator.classes,CNN_prediction_final , target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
# Get the confusion matrix
CM = confusion_matrix(test_generator.classes, CNN_prediction_final)

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.xticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.yticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.show()
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import auc

fpr , tpr , thresholds = roc_curve ( test_generator.classes , predictions)
auc_keras = auc(fpr, tpr)
print("AUC Score:",auc_keras)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_keras)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
model_feat = tf.keras.Model(inputs=cnn_model.input,outputs=cnn_model.get_layer('dense_1').output)
feat_train = model_feat.predict_generator(train_generator)
feat_test = model_feat.predict_generator(test_generator)

df = pd.DataFrame(feat_train)
df.shape
from sklearn.svm import SVC
from sklearn import metrics
svm = SVC(kernel='sigmoid')
svm.fit(feat_train,train_generator.classes)
svmpredict = svm.predict(feat_test)
print(classification_report(test_generator.classes,svmpredict , target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
fpr , tpr , thresholds = roc_curve ( test_generator.classes ,svmpredict)
auc_keras = auc(fpr, tpr)
print("AUC Score:",auc_keras)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_keras)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
svmCM = confusion_matrix(test_generator.classes, svmpredict)

fig, ax = plot_confusion_matrix(conf_mat=svmCM ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.xticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.yticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(feat_train,train_generator.classes)

knnpredict = knn.predict(feat_test)
print(classification_report(test_generator.classes,knnpredict , target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
knn.score(feat_test,test_generator.classes)
fpr , tpr , thresholds = roc_curve ( test_generator.classes ,knnpredict)
auc_keras = auc(fpr, tpr)
print("AUC Score:",auc_keras)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_keras)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show() 
# Get the confusion matrix
knnCM = confusion_matrix(test_generator.classes, knnpredict)

fig, ax = plot_confusion_matrix(conf_mat=knnCM ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.xticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.yticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.show()
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(feat_train,train_generator.classes)

dtpredict = dt.predict(feat_test)
print(classification_report(test_generator.classes,dtpredict , target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
fpr , tpr , thresholds = roc_curve ( test_generator.classes ,dtpredict)
auc_keras = auc(fpr, tpr)
print("AUC Score:",auc_keras)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_keras)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
IMG_SIZ = 64 
BATCH_SIZ = 16
train_generator_ = img_generator.flow_from_directory(path_train,
                                                  target_size=(IMG_SIZ, IMG_SIZ),
                                                  batch_size=BATCH_SIZ,
                                                  color_mode="grayscale",
                                                  shuffle=True,
                                                  class_mode='binary', 
                                                  subset='training')

val_generator_  = img_generator.flow_from_directory(path_val,
                                                  target_size=(IMG_SIZ, IMG_SIZ),
                                                  color_mode="grayscale",
                                                  class_mode='binary',
                                                  )

test_generator_ = img_generator.flow_from_directory(path_test,
                                                  target_size=(IMG_SIZ, IMG_SIZ),
                                                  color_mode="grayscale",
                                                  class_mode='binary')
cnn2_model = Sequential()
cnn2_model.add(Flatten(input_shape=(IMG_SIZ, IMG_SIZ,1)))
cnn2_model.add(Dense(512 ,activation='relu'))
cnn2_model.add(Dense(256, activation='relu'))
cnn2_model.add(Dense(1, activation='sigmoid'))


cnn2_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

cnn2_model.summary()
model = cnn2_model.fit_generator(train_generator_,
                                steps_per_epoch = 163,
                                epochs = 10,
                                validation_data=val_generator_,
                                callbacks = callback)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(model.history['val_accuracy'],label='validation Accuracy')
plt.plot(model.history['accuracy'], label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training  Accuracy')

plt.subplot(1, 2, 2)
plt.plot(model.history['val_loss'],label='validation loss')
plt.plot(model.history['loss'], label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training  Loss')

plt.show()
# evaluate model
predic2= cnn2_model.evaluate(test_generator_)
predictions2 =  cnn2_model.predict_generator(test_generator_)
CNN_prediction_final_ = np.where(predictions2>0.5,1,0)
print(classification_report(test_generator_.classes,CNN_prediction_final_ , target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
# Get the confusion matrix
CM2 = confusion_matrix(test_generator_.classes, CNN_prediction_final_)

fig, ax = plot_confusion_matrix(conf_mat=CM2 ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.xticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.yticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.show()
fpr , tpr , thresholds = roc_curve (test_generator.classes ,predictions2)
auc_keras = auc(fpr, tpr)
print("AUC Score:",auc_keras)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc_keras)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()