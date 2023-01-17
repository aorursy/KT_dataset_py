import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
labels = ['PNEUMONIA', 'NORMAL']
img_size = 184
def datafunc(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.cvtColor(img_arr,cv2.COLOR_GRAY2RGB)               
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
train = datafunc('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')
test = datafunc('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
#val = datafunc('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')
trainlabel = []
for img in train:
    if(img[1] == 0):
        trainlabel.append("Pneumonia")
    else:
        trainlabel.append("Normal")
sns.countplot(trainlabel)        
plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
#for feature, label in val:
 #   x_val.append(feature)
  #  y_val.append(label)
x_train = np.array(x_train)/255.0
x_test = np.array(x_test)/255.0
#x_val = np.array(x_val)/255.0
x_train = (x_train.reshape(-1,img_size,img_size,3))
x_test = (x_test.reshape(-1,img_size,img_size,3))
#x_val = (x_val.reshape(-1,img_size,img_size,3))
x_train.shape
y_train=np.array(y_train)
y_test=np.array(y_test)
#y_val=np.array(y_val)
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
inputs = (184,184,3)

base_model = keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=inputs)
base_model.trainable = False

x = base_model.output
x= GlobalAveragePooling2D()(x)
x= Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=BatchNormalization()(x)
x= Dense(512,activation='relu')(x)
x=Dropout(0.4)(x)
predictions = Dense(2, activation='softmax')(x)
for layer in base_model.layers:
    layer.trainable = False

# this is the model we will train
model=Sequential()
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='Adam',
              metrics=['accuracy'])
datagen= ImageDataGenerator()
datagen.fit(x_train)
from keras.callbacks import ModelCheckpoint, EarlyStopping
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 2 , validation_data = datagen.flow(x_test, y_test))
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
for layer in model.layers[:537]:
    layer.trainable = False
for layer in model.layers[537:]:
    layer.trainable = True
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("DenseNet.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 2 , validation_data = datagen.flow(x_test, y_test),callbacks=[checkpoint,early])
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = model.predict(x_test)
predictions = predictions[:,0]
i=0
for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i]=0
    else:
        predictions[i]=1

cm = confusion_matrix(y_test,predictions)
cm
from sklearn.metrics import precision_score , recall_score
print(precision_score(y_test,predictions,average=None))
print(recall_score(y_test,predictions,average=None))

from sklearn.metrics import roc_curve,roc_auc_score
fpr , tpr , thresholds = roc_curve ( y_test , predictions)

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
plot_roc_curve (fpr,tpr) 

auc_score=roc_auc_score(y_test , predictions)  
print(auc_score)
