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
# Importando as Bibiotecas necessárias: 

import pandas as pd
import numpy as np
import cv2    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

plt.style.use('ggplot')

%matplotlib inline
# Verificando a versão do Tensorflow
import tensorflow as tf
print(tf.__version__)
# Definindo variáveis: 
pasta_principal = '../input/celeba-dataset/'
pasta_imagens = pasta_principal + 'img_align_celeba/img_align_celeba/'

EXEMPLO = pasta_imagens + '001150.jpg'

AMOSTRA_TREINO = 7000
AMOSTRA_VALIDACAO = 1500
AMOSTRA_TESTE = 1000
LARGURA = 178
ALTURA = 218
BATCH_SIZE = 16
#EPOCAS = 20
# carregando o arquivo com os atributos de cada imagem:
df_attr = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')
df_attr.head()
# transformando a columa com o nome da imagem em índice
df_attr.set_index('image_id', inplace=True)
# alterando o valor de -1 (quando a imagem nao apresenta o atributo), para 0
df_attr.replace(to_replace=-1, value=0, inplace=True) 
# verificando as dimensões do dataser
df_attr.shape
# Listando as colunas com os atributoss
df_attr.columns
# Carregando um exemplo de imagem:
img = load_img(EXEMPLO)
plt.grid(False)
plt.imshow(img)
df_attr.loc[EXEMPLO.split('/')[-1]][['Smiling','Male',"Young"]]
# Carregando o dataset com a partição em dados de treino, validação e teste:
df_partition = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')
df_partition.head()

# Verificando a quantidade de observações em cada tipo de partição:
df_partition['partition'].value_counts().sort_index()
# Criando um dataset com o tipo de partição e a variável Target 'Male'
df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(df_attr['Male'], how='inner')
df_par_attr.head()
# Verificando o balanceamento das classes:
plt.title('Proporção de imagens masculinas e femininas no dataset CelebA')
sns.countplot(y='Male', data=df_attr, color="b")
plt.show()
# Definindo a função que fará a transformação necessária nas imagens:

def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x

# Definindo a função que fará o balanceamento das classes (há mais imagens do sexo feminino)

def generate_df(partition, attr, num_samples):
    '''
    partição:
        0 -> treino
        1 -> validação
        2 -> teste
    
    '''
    
    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(pasta_imagens + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np.array(df_[attr])
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(pasta_imagens + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (LARGURA, ALTURA)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_
# Gerando os dados de Treino com Balanceamento
x_train, y_train = generate_df(0, 'Male', AMOSTRA_TREINO) 

# Pré tratamento dos dados de treino com Data Augmentation
# Data Augmentation permite gerar imagens diferentes das originais, permitindo que o modelo aprenda
# com estas variações (mudança no ângulo, tamanho e posição)

train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
x_train, y_train,
batch_size=BATCH_SIZE,
)

# Verificando o shape dos dados de treino
print(f's_train.shape = {x_train.shape}')
print(f's_train.shape = {y_train.shape}')
# Gerando os dados de validação com balanceamento
x_valid, y_valid = generate_df(1, 'Male', AMOSTRA_VALIDACAO)

# Construindo o modelo:
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides = 2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides = 2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides = 2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(2, activation='softmax')(x)

model = Model(i, x)
# Compilamos o modelo usando o optimzador Adam
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fazendo o fit do modelo:
r = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5)
# Gráfico do loss do modelo:
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='treino')
plt.plot(r.history['val_loss'], label='validação')
plt.legend()
# Gráfico da acurácia:
plt.plot(r.history['accuracy'], label='treino')
plt.plot(r.history['val_accuracy'], label='validação')
plt.legend()
# Gerando os dados de Teste com balanceamento
x_test, y_test = generate_df(2, 'Male', AMOSTRA_TESTE)
# Calculando a acurácia do modelo nos dados de teste: 
from sklearn.metrics import f1_score

# generate prediction
model_predictions = [np.argmax(model.predict(feature)) for feature in x_test ]

# report test accuracy
test_accuracy = 100 * np.sum(np.array(model_predictions)==y_test) / len(model_predictions)
print('Model Evaluation')
print('Test accuracy: %.4f%%' % test_accuracy)
print('f1_score:', f1_score(y_test, model_predictions))
