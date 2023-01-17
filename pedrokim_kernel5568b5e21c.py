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


#imports
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend
from PIL import Image
## Rodar apenas uma vez no Kaggle se acontecer alguma coisa é necessário reiniciar o kernel
os.makedirs('greyscale_treino')
os.makedirs('greyscale_treino/benigno')
os.makedirs('greyscale_treino/maligno')
os.makedirs('greyscale_teste')
os.makedirs('greyscale_teste/benigno')
os.makedirs('greyscale_teste/maligno')
    
    
#### Criação de versões em preto e branco dos dados de treino

import os
dir_color = '../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_treino/benigno'
#DIR2= '../input/training_child/training_child'
print('Inicio ... convertendo treino Benigno ...')
i = 1
for img_address in os.listdir(dir_color):
    img = Image.open(dir_color+'/'+img_address).convert('LA')
    #print(i)
    save_dir = 'greyscale_treino/benigno/img'+str(i)+'.png'
    img.save(save_dir)
    i=i+1

print('... agora treino Maligno ...')
dir_color = '../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_treino/maligno'
i = 1
for img_address in os.listdir(dir_color):
    img = Image.open(dir_color+'/'+img_address).convert('LA')
    #print(i)
    save_dir = 'greyscale_treino/maligno/img'+str(i)+'.png'
    img.save(save_dir)
    i=i+1

print('... paciência agora o teste Benigno ...')
dir_color = '../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste/benigno'
i = 1
for img_address in os.listdir(dir_color):
    img = Image.open(dir_color+'/'+img_address).convert('LA')
    #print(i)
    save_dir = 'greyscale_teste/benigno/img'+str(i)+'.png'
    img.save(save_dir)
    i=i+1

print('... não crashou  agora vem o teste Maligno ...')
dir_color = '../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste/maligno'
i = 1
for img_address in os.listdir(dir_color):
    img = Image.open(dir_color+'/'+img_address).convert('LA')
    #print(i)
    save_dir = 'greyscale_teste/maligno/img'+str(i)+'.png'
    img.save(save_dir)
    i=i+1


    
print('Fim')
# criando métrica recall
def recall_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall
# criando métrica precisão
def precision_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision
# criando métrica f1
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))
# Inicializando a Rede Neural Convolucional
classifier = Sequential()
# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Passo 3 - Flattening
classifier.add(Flatten())
# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',f1_m,precision_m, recall_m])
# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)
# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory('../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_treino',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')
# Executando o treinamento
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

classifier.fit_generator(training_set,
                         epochs = 5,            
                         validation_data = validation_set
                        )
# salvar o modelo treinado
classifier.save('../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/modelo.h5')
# carregar o modelo treinado
classifier = keras.models.load_model('../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/modelo.h5', custom_objects={'f1_m': f1_m, 'recall_m': recall_m, 'precision_m': precision_m})
from IPython.display import Image
# Avaliando Primeira Imagem
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste/benigno/ISIC_0000260.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 0:
    prediction = 'benigno'
else:
    prediction = 'maligno'

Image(filename='../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste/benigno/ISIC_0000260.jpg')
prediction
# Avaliando Segunda Imagem
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste/maligno/ISIC_0009894.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 0:
    prediction = 'benigno'
else:
    prediction = 'maligno'

Image(filename='../input/deteco-de-melanomas-em-imagens-dermatoscpicas/TCC v1/dataset_teste/maligno/ISIC_0009894.jpg')
prediction