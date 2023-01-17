

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = "../input/flowers-recognition/flowers/"
folders = os.listdir(data)
print(folders)

import cv2
from tqdm import tqdm
image_names = []
train_labels = []
train_images = []

size = 120,120

for folder in folders:
    for file in tqdm(os.listdir(os.path.join(data,folder))):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue


X1 = np.array(train_images)

X1.shape
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le=LabelEncoder()
Y=le.fit_transform(train_labels)
Y=to_categorical(Y,5)
X=np.array(X1)
X=X/255
print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.20, random_state=42)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

import matplotlib.pyplot as plt
import seaborn as sns
for i in range(10,100,20):
    plt.imshow(X_train[i][:,:,0],cmap='gray')
    plt.show()
del X,Y
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'valid', 
                 activation ='relu', input_shape = (120,120,3)))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters =128, kernel_size = (3,3),padding = 'valid', 
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Conv2D(filters = 160, kernel_size = (3,3),padding = 'valid',
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 192, kernel_size = (3,3),padding = 'valid',
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'valid',
                 activation ='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2)) 

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(5, activation = "softmax"))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 50
batch_size = 256

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False, 
        rotation_range=10, 
        zoom_range = 0.8,
        width_shift_range=0.8,  
        height_shift_range=0.8,  
        horizontal_flip=False,  
        vertical_flip=False)  

datagen.fit(X_train) 
model.summary()

%%time
history = model.fit(X_train,Y_train, batch_size=batch_size, epochs = epochs, validation_data = (X_val,Y_val))
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json


from keras import layers
from keras import models



def guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos):
    print("Guardando Red Neuronal en Archivo")  
    # serializar modelo a JSON

    # Guardar los Pesos (weights)
    model.save_weights(nombreArchivoPesos+'.h5')

    # Guardar la Arquitectura del modelo
    with open(nombreArchivoModelo+'.json', 'w') as f:
        f.write(model.to_json())

    print("Red Neuronal Grabada en Archivo")   
    
def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):
        
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo+'.json', 'r') as f:
        model = model_from_json(f.read())

    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos+'.h5')  

    print("Red Neuronal Cargada desde Archivo") 
    return model

model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=0)
model.summary()
print('Resultado en Train:')
score = model.evaluate(X_train, Y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Fase de Testing
print('Resultado en Test:')
score = model.evaluate(X_val, Y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

nombreArchivoModelo='arquitectura_base'
nombreArchivoPesos='pesos_base'
guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos)
!pip install -U git+https://github.com/qubvel/efficientnet
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet201
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.utils import Sequence
import efficientnet.keras as efn 
def get_model():
    base_model =  efn.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg', input_shape=(120, 120, 3))
    x = base_model.output
    y_pred = Dense(5, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=y_pred)

model = get_model()
!pip install keras_radam
from keras_radam import RAdam
model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5),  loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10
batch_size = 256
history_0 = model.fit(X_train,Y_train, batch_size=batch_size, epochs = epochs, validation_data = (X_val,Y_val),
                              verbose=1
                             )

score = model.evaluate(X_train, Y_train, verbose=0)
print('Resultado en Train:')
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Fase de Testing
print('Resultado en Test:')
score = model.evaluate(X_val, Y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

nombreArchivoModelo='arquitectura_optimizada'
nombreArchivoPesos='pesos_optimizados'
guardarRNN(model,nombreArchivoModelo,nombreArchivoPesos)
from sklearn.metrics import confusion_matrix
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_val,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Reds",linecolor="gray", fmt= '.2f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import gc
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.plot(history_0.history['acc'][1:])
plt.plot(history_0.history['val_acc'][1:])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')

plt.title('model accuracy')

plt.subplot(1,2,2)
plt.plot(history_0.history['loss'][1:])
plt.plot(history_0.history['val_loss'][1:])
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.title('model loss')
gc.collect()
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

test_dir = "../input/flowers/flowers/rose/10090824183_d02c613f10_m.jpg"
def convert_image_to_array(file):
    return img_to_array(load_img(file))

X = np.array(convert_image_to_array(test_dir))
X=X.astype('float32')
print("Imagen de testeo", X.shape)


def cvtRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
img_width, img_height = 120, 120

def predict_one_image(img, model):
  img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
  img = np.reshape(img, (1, img_width, img_height, 3))
  img = img/255.
  pred = model.predict(img)
  class_num = np.argmax(pred)
  return class_num, np.max(pred)
# idx = 120
# pred, probability = predict_one_image(images[4][idx], model_ResNet50)
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
test_img = cv2.imread("../input/flowers/flowers/rose/10090824183_d02c613f10_m.jpg")
pred, probability = predict_one_image(test_img, model)
print('%s %d%%' % (categories[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(cvtRGB(test_img))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()
# idx = 120
# pred, probability = predict_one_image(images[4][idx], model_ResNet50)

test_img = cv2.imread('flowers-recognition/flowers/rose/10090824183_d02c613f10_m.jpg')
pred, probability = predict_one_image(test_img, model)
print('%s %d%%' % (categories[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(cvtRGB(test_img))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()