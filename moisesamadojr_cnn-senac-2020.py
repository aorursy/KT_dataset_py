import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16





import numpy as np
## Descompactando os arquivos zipados

import zipfile

with zipfile.ZipFile("../input/aerial-cactus-identification/train.zip","r") as z:

    z.extractall(".")



with zipfile.ZipFile("../input/aerial-cactus-identification/test.zip","r") as z:

    z.extractall(".")  
## Atribuindo os arquivos e diretórios a variáveis para futuras análises



train_dir='train'

test_dir='test'

train=pd.read_csv('../input/aerial-cactus-identification/train.csv')



test_df=pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
#Localizando GPU

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import tensorflow as tf
# Imprimindo uma amostra do nosso dataset de treino

train.head(5)
# Conversão da coluna 'has_cactus' em string para realizar a separação dos dados

train.has_cactus=train.has_cactus.astype(str)

#Quantidade de linhas e colunas do df

train.shape
#Quantos registros são cactus e quantos nao são

train['has_cactus'].value_counts()
#Transforma os dados de entrada / utilizado para Augmentation



#Rescale: Altera o dimensionamento da imagem, os dados originais são multiplixados pelo valor setado

#rotation_range: Rotação de 20 graus nas imagens, 

#horizontal_flip: Reverter linhas <->

#shear_range: Estica a imagem

datagen=ImageDataGenerator(rescale=1./255, rotation_range=20,horizontal_flip=True, shear_range = 0.2,zoom_range = 0.2)

#Leitura dos dados presentes no diretório Train_dir de acordo com o dataframe de treino, subdividindo em treino e validação

train_generator=datagen.flow_from_dataframe(dataframe=train[:15001],directory=train_dir,x_col='id',

                                            y_col='has_cactus',class_mode='binary',

                                            target_size=(32,32))





validation_generator=datagen.flow_from_dataframe(dataframe=train[15001:],directory=train_dir,x_col='id',

                                                y_col='has_cactus',class_mode='binary',

                                                target_size=(32,32))



#CNN sem utilização de tranfer learning

with tf.device('/GPU:0'):

    model=models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape = (32,32,3)))

    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64,(3,3),activation='relu', input_shape = (32,32,3)))

    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation='relu', input_shape = (32,32,3)))

    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512,activation='relu'))

    model.add(layers.Dense(1,activation='sigmoid'))
#Verificaçao da arquitetura da CNN:

model.summary()
# Compilando a CNN

model.compile(loss='binary_crossentropy',optimizer='Adamax',metrics=['acc'])
# Executando o treinamento (sem VGG16)

epochs=10

history=model.fit_generator(train_generator,steps_per_epoch=450,epochs=10,validation_data=validation_generator,validation_steps=4500)


# Plotar a acurácia/desempenho do modelo de acordo com as épocas, comparando traino da validação

fig = plt.figure(figsize=(12,8))

plt.plot(history.history['acc'],'blue')

plt.plot(history.history['val_acc'],'orange')

plt.xticks(np.arange(0, 10, 1))

plt.yticks(np.arange(0.8,1.1,.05))

plt.rcParams['figure.figsize'] = (10, 10)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.grid(True)

plt.gray()

plt.legend(['train','validation'])

plt.show()

 

plt.figure(1)

plt.plot(history.history['loss'],'blue')

plt.plot(history.history['val_loss'],'orange')

plt.xticks(np.arange(0, 10, 1))

plt.rcParams['figure.figsize'] = (10, 10)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.grid(True)

plt.gray()

plt.legend(['train','validation'])

plt.show()
#melhorando o modelo utilizando o VGG16 

#include_top = False: carregar VGG sem a parte classificadora do modelo

model_vg=VGG16(weights='imagenet',include_top=False,input_shape=(32, 32, 3))

model_vg.summary()
# Não altera os pesos da VGG

model_vg.trainable = False
#CNN com utilização de transfer learning

from keras.layers import Activation, Dropout, Flatten, Dense,Conv2D,Conv3D,MaxPooling2D,AveragePooling2D,BatchNormalization

model2 = models.Sequential()

model2.add(model_vg)



model2.add(Flatten())

model2.add(Dense(256, use_bias=True))

model2.add(BatchNormalization())

model2.add(Activation("relu"))

model2.add(Dropout(0.5))

model2.add(Dense(64,activation='relu'))

model2.add(BatchNormalization())

model2.add(Dense(16, activation='tanh'))

model2.add(Dense(1, activation='sigmoid'))







# Compilando a rede

 

with tf.device('/GPU:0'):

    model2.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['acc'])

# steps_per_epoch: Quantas amostras são extraídas do gerador para prosseguir para a próxima época.Após a extração do lote de amostras do gerador(ou seja, depois de executar as etapas do gradiente descendente que procura localizar o mínimo global da funcao) inicia-se a próxima época.

# função de custa 

# Executando o treinamento 

#loss(baseado no treino):valor da função de custo

#val_loss(Baseado na validação):Se Val_loss for significamente maior, significa que houve overfiting no treinamento

with tf.device('/GPU:0'):

    history_vgg=model2.fit_generator(train_generator,steps_per_epoch=450,epochs=15,validation_data=validation_generator,validation_steps=450)

# Plotar a acurácia/desempenho do modelo de acordo com as épocas, comparando traino da validação

fig = plt.figure(figsize=(12,8))

plt.plot(history_vgg.history['acc'],'blue')

plt.plot(history_vgg.history['val_acc'],'orange')

plt.xticks(np.arange(0, 15, 1))

plt.yticks(np.arange(0.8,1.1,.05))

plt.rcParams['figure.figsize'] = (10, 10)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.grid(True)

plt.gray()

plt.legend(['train','validation'])

plt.show()

 

plt.figure(1)

plt.plot(history_vgg.history['loss'],'blue')

plt.plot(history_vgg.history['val_loss'],'orange')

plt.xticks(np.arange(0, 15, 1))

plt.rcParams['figure.figsize'] = (10, 10)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.grid(True)

plt.gray()

plt.legend(['train','validation'])

plt.show()
#exportar o resultado:

test_dir='test/'



X_test = []

imges = test_df['id'].values



for img_id in imges:

    X_test.append(cv2.imread(test_dir + img_id )) 

                  

X_test = np.asarray(X_test)

X_test = X_test.astype('float32')

X_test /= 255



y_test_pred  = model.predict_proba(X_test)



test_df['has_cactus'] = y_test_pred

test_df.to_csv('submission.csv', index=False)