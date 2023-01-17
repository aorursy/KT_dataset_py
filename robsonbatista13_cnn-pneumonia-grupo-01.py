import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline

import keras
from keras.models import Sequential,Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense, Input,SeparableConv2D
from keras.layers import Dropout, BatchNormalization, ZeroPadding2D
from keras.models import load_model
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve,auc
from mlxtend.plotting import plot_confusion_matrix
from keras.utils import to_categorical
from glob import glob
# Importando os dados
train =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
test  =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test'
val   =  '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'
imgNormal = glob(train+"/NORMAL/*.jpeg")
imgNormal = np.asarray(plt.imread(imgNormal[1]))
imgPneumonia = glob(train+"/PNEUMONIA/*.jpeg")
imgPneumonia = np.asarray(plt.imread(imgPneumonia[5]))

f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(imgNormal)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(imgPneumonia)
a2.set_title('Pneumonia')
#O batch_size define a quantidade de imagens que serão lidas por vez.

batch_size = 16

#Aqui definimos as transformações que serão aplicadas nas imagens.

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   width_shift_range=0.10,
                                   height_shift_range=0.10,
                                   rotation_range=20,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   vertical_flip=False,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255) 


training_set = train_datagen.flow_from_directory(train,
                                                 target_size = (128, 128),                                                 
                                                 color_mode='rgb',
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(val,
                                                        target_size=(128, 128),
                                                        batch_size=batch_size,
                                                        color_mode='rgb',
                                                        shuffle=False,
                                                        class_mode='binary')

test_set = test_datagen.flow_from_directory(test,
                                            target_size = (128, 128),
                                            color_mode='rgb',
                                            shuffle=False,                                            
                                            batch_size = batch_size,
                                            class_mode = 'binary')


def construcao_modelo(shape=(128,128,3)):

    modelo = Sequential()

    #Primeira camanda
    modelo.add(Conv2D(32, (4, 4), activation="relu", input_shape=shape))

    #Definindo o MaxPooling
    modelo.add(MaxPooling2D(pool_size = (4, 4)))
    
    #Camada para tratar o overfitting,nesse caso em cada epochs zeraremos 30% dos neurônios
    modelo.add(Dropout(0.3))
    
    #Segunda camada
    modelo.add(Conv2D(32, (4, 4), activation="relu"))
    modelo.add(MaxPooling2D(pool_size = (2, 2)))
    
    #Tradando overfitting na segunda camada
    modelo.add(Dropout(0.3))
    
    # Da um reshape no output transformando em array
    modelo.add(Flatten())

    # Camada Dense
    modelo.add(Dense(128, activation = 'relu'))
    
    #Camada para tratar o overfitting
    modelo.add(Dropout(0.5))
    
    #Camada de saida
    modelo.add(Dense(1, activation = 'sigmoid'))
    
    return modelo
#resultado da arquitetura do modelo
modelo = construcao_modelo()
modelo.summary()
modelo.compile(optimizer = 'adam',
               loss = 'binary_crossentropy',
               metrics = ['accuracy'])
filepath = 'melhor_modelo.hdf5'

checkpoint = ModelCheckpoint(filepath=filepath, 
                            monitor='val_loss', 
                            verbose=1, mode='min', 
                            save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss',
                            min_delta=0.001,
                            patience=5,
                            mode='min',
                            verbose=1)

lr_reduce = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.3,
                              patience=2,
                              verbose=2,
                              mode='auto')

r = modelo.fit_generator(training_set,
                         epochs = 50,
                         validation_data = validation_generator,
                         callbacks=[checkpoint, early_stop,lr_reduce]
                        )

#carregando o melhor modelo
melhorModelo = load_model('melhor_modelo.hdf5')
# Acurácia e loss do modelo
loss, acuracia = melhorModelo.evaluate_generator(test_set)
print("Loss: %.4f" % (loss))
print("Acurácia: %.2f%%" % (acuracia*100))
fontsize = 15
plt.style.use("_classic_test_patch")
plt.figure(figsize=(12,8))
plt.plot(r.history["loss"], label="train_loss")
plt.plot( r.history["val_loss"], label="val_loss")
plt.plot( r.history["accuracy"], label="train_acc")
plt.plot( r.history["val_accuracy"], label="val_acc")
plt.title("Comparação do desempenho do modelo em train e validação \n",fontsize=fontsize)
plt.xlabel("Epoch",fontsize=fontsize)
plt.ylabel("Loss / Acurácia",fontsize=fontsize)
plt.legend(loc="best")
pred = modelo.predict_generator(test_set)
pred = pred > 0.5
cm  = confusion_matrix(test_set.classes,pred)

plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Teste",fontsize=fontsize)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=fontsize)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=fontsize)
plt.show()
predValidacao = melhorModelo.predict_generator(validation_generator)
predValidacao = predValidacao > 0.5
cm  = confusion_matrix(validation_generator.classes,predValidacao)

plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Validação",fontsize=fontsize)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=fontsize)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=fontsize)
plt.show()
fpr, tpr, threshold = roc_curve(test_set.classes, pred)
roc_auc = auc(fpr, tpr)
X = [[0,1],[0,1]]

plt.figure(figsize=(12,8))
plt.title('Curva ROC',fontsize=fontsize)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot(X[0],X[1],'r--')
plt.xlim(X[0])
plt.ylim(X[0])
plt.ylabel('Verdadeiros Positivos',fontsize=fontsize)
plt.xlabel('Falsos Positivos',fontsize=fontsize)
plt.show()
fpr, tpr, threshold = roc_curve(validation_generator.classes, predValidacao)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(12,8))
X = [[0,1],[0,1]]
plt.title('Curva ROC - Validação',fontsize=fontsize)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot(X[0],X[1],'r--')
plt.xlim(X[0])
plt.ylim(X[0])
plt.ylabel('Verdadeiros Positivos',fontsize=fontsize)
plt.xlabel('Falsos Positivos',fontsize=fontsize)
plt.show()
print(classification_report(test_set.classes,pred))