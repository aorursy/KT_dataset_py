!pip install keras-tuner
!nvidia-smi

import numpy as np 
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt
import random
from functools import partial
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kerastuner as kt
from kerastuner.engine.hyperparameters import HyperParameters
from keras.callbacks import ReduceLROnPlateau
import shutil
from keras.regularizers import l2
from sklearn.metrics import classification_report,confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix,roc_curve, auc,classification_report
SHAPE_OF_IMAGES = (150,150)
train_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train'
test_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test'
val_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val'
f = plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.bar(x=["pneumonia","normal"], height = (len(os.listdir(train_dir + "/PNEUMONIA")),len(os.listdir(train_dir + "/NORMAL"))))
plt.title("Training data set")
plt.subplot(1,3,2)
plt.bar(x=["pneumonia","normal"], height = (len(os.listdir(test_dir + "/PNEUMONIA")),len(os.listdir(test_dir + "/NORMAL"))))
plt.title("Test data set")
plt.subplot(1,3,3)
plt.bar(x=["pneumonia","normal"], height = (len(os.listdir(val_dir + "/PNEUMONIA")),len(os.listdir(val_dir + "/NORMAL"))))
plt.title("Validation data set")
def view_images(directory,n_of_images):
    plt.figure(figsize=(25,10))
    for i in range(0,n_of_images,2):
        random_label_p = random.choice(range(len(os.listdir(directory + "/PNEUMONIA"))))
        random_label_n = random.choice(range(len(os.listdir(directory + "/NORMAL"))))
        plt.subplot(2,n_of_images // 2, i+1)
        plt.imshow(cv2.imread(os.path.join(directory + "/PNEUMONIA/",os.listdir(directory + "/PNEUMONIA")[random_label_p])))
        plt.title("PNEUMONIA")
        plt.subplot(2,n_of_images // 2, i+2)
        plt.imshow(cv2.imread(os.path.join(directory + "/NORMAL/",os.listdir(directory + "/NORMAL")[random_label_n])))
        plt.title("NORMAL")
        plt.axis("off")
    plt.show()
view_images(train_dir,16)
train_data_generator = ImageDataGenerator(
   width_shift_range=0.2,
   height_shift_range=0.2,
   rescale=1./255,
   zoom_range=0.1,
   vertical_flip = True)
val_data_generator = ImageDataGenerator(
   width_shift_range=0.2,
   height_shift_range=0.2,
   rescale=1./255,
   zoom_range=0.1,
   vertical_flip = True)
test_data_generator = ImageDataGenerator(rescale=1./255)
train_generator = train_data_generator.flow_from_directory(
        train_dir,
        target_size=SHAPE_OF_IMAGES,
        color_mode="grayscale",
        batch_size=64,
        class_mode='binary')
validation_generator = val_data_generator.flow_from_directory(
        val_dir,
        color_mode="grayscale",
        target_size=SHAPE_OF_IMAGES,
        batch_size=32,
        class_mode='binary')
test_generator = test_data_generator.flow_from_directory(
        test_dir,
        color_mode="grayscale",
        target_size=SHAPE_OF_IMAGES,
        batch_size=64,
        class_mode='binary')

model = Sequential([
        Conv2D(filters=32,kernel_size=3,activation = 'relu',padding = 'same', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005),input_shape = (150,150,1)),
        Dropout(0.25),
        BatchNormalization(),
        MaxPooling2D(padding= 'same'),
        Conv2D(filters=64,kernel_size=3,activation = 'relu',padding = 'same', kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005)),
        Dropout(0.25),
        BatchNormalization(),
        MaxPooling2D(padding= 'same'),
        Flatten(),
        Dense(units = 128,activation= 'relu', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)),
        Dropout(0.25),
        BatchNormalization(),
        Dense(1,activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss = 'binary_crossentropy',metrics=['accuracy'])

model.summary()
history = model.fit(train_generator ,epochs = 50, verbose = 1,validation_data=validation_generator)
model.evaluate(test_generator)
model = load_model('../input/model-for-xray/savedmodel.h5')
plt.figure(figsize=(25,10))
plt.plot(range(1,51),model.history.history['accuracy'],color="blue", label="Training accuracy", linestyle="-")
plt.plot(range(1,51),model.history.history['val_accuracy'],color="red", label="Validation accuracy", linestyle="-")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Training accuracy VS Validation accuracy")
plt.legend()
plt.show()
plt.figure(figsize=(25,10))
plt.plot(range(1,51),history.history['loss'],color="blue", label="Training loss", linestyle="-")
plt.plot(range(1,51),history.history['val_loss'],color="red", label="Validation loss", linestyle="-")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training loss VS Validation loss")
plt.legend()
plt.show()
def get_test_set(directory):
  data = []
  y = []
  for label in ['NORMAL','PNEUMONIA']:
    path = os.path.join(directory, label)
    class_num = ['NORMAL','PNEUMONIA'].index(label)
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, SHAPE_OF_IMAGES) 
            data.append(resized_arr)
            y.append(class_num)
        except Exception as e:
            print(e)
  return np.array(data),np.array(y)

X_test,y_test = get_test_set(test_dir)
X_test = X_test / 255.
X_test = X_test.reshape(-1,150,150,1)
y_pred = model.predict_classes(X_test)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
cm
import seaborn as sns
plt.figure(figsize=(20,15))
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='',xticklabels=['NORMAL','PNEUMONIA'],yticklabels=['NORMAL','PNEUMONIA'])
def plot_roc_auc_curve(model,X,y):
  
  y_score = model.predict_classes(X)

  fpr, tpr, thresholds = roc_curve(y, y_score)

  fig = px.area(
      x=fpr, y=tpr,
      title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
      labels=dict(x='False Positive Rate', y='True Positive Rate'),
      width=1400, height=700
  )
  fig.add_shape(
      type='line', line=dict(dash='dash'),
      x0=0, x1=1, y0=0, y1=1
  )

  fig.update_yaxes(scaleanchor="x", scaleratio=1)
  fig.update_xaxes(constrain='domain')
  
  fig.show()
plot_roc_auc_curve(model,X_test,y_test)