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
# Importanção do pandas
import pandas as pd
# Importanção do numpy
import numpy as pd
# Importação do matplotlib
import matplotlib.pyplot as plt
# Importação do seaborn
import seaborn as sns
# Importação tensorflow

#!pip install tensorflow as tf

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model
# Importação do keras

# !pip install keras

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras import optimizers

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
# Importação geral
from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64
%matplotlib inline
# Verificação dos dados
diretorio = '../input/celeba-dataset/'
imagens = diretorio + 'img_align_celeba/img_align_celeba/'
# Atribuição de valores
EXAMPLE_PIC = imagens + '000034.jpg'

TRAINING_SAMPLES = 5000

VALIDATION_SAMPLES = 200

TEST_SAMPLES = 200

IMG_WIDTH = 178

IMG_HEIGHT = 218

BATCH_SIZE = 16

NUM_EPOCHS = 20
# Importação do pandas
import pandas as pd
# Obtenção dos dados e atributos das figuras
dataframe_dados = pd.read_csv(diretorio + 'list_attr_celeba.csv')
dataframe_dados.set_index('image_id', inplace = True)
dataframe_dados.replace(to_replace = -1, value = 0, inplace = True)
# Análise da base
dataframe_dados.shape
dataframe_dados.info()
# Visualização de uma imagem
img = load_img(EXAMPLE_PIC)
plt.grid(False)
plt.imshow(img)
dataframe_dados.loc[EXAMPLE_PIC.split('/')[-1]][['Smiling','Male','Young']] #some attributes
# Análise da base por sexo
plt.title('Mulher ou Homem')
sns.countplot(y = 'Male', data = dataframe_dados, color = "r")
plt.show()
# Início do estudo
# Base de validação, treino e teste
dataframe_part = pd.read_csv(diretorio + 'list_eval_partition.csv')
dataframe_part.head()
dataframe_part['partition'].value_counts().sort_index()
# De acordo com o index 0, 1 e 2 são treino, validação e teste respectivamente.
dataframe_part.set_index('image_id', inplace=True)
dataframe_partic_atributo = dataframe_part.join(dataframe_dados['Male'], how='inner')
dataframe_partic_atributo.head()
# Splitting a base
# Realização do balanceamento da base e criação das partições

# Criação da função
def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''
    

    df_ = dataframe_partic_atributo[(dataframe_partic_atributo['partition'] == partition) & (dataframe_partic_atributo[attr] == 0)].sample(int(num_samples / 2))
    
    df_ = pd.concat([df_,dataframe_partic_atributo[(dataframe_partic_atributo['partition'] == partition) & (dataframe_partic_atributo[attr] == 1)].sample(int(num_samples / 2))])

    
    # Base de Treino e Validação
    if partition != 2:
        x_ = np.array([load_reshape_img(imagens + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0],218,178,3)
        y_ = df_[attr].values
        
    # Base de Teste
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(imagens + index)
            im = cv2.resize(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis = 0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_

# Processamento das imagens, aprendendo as variações do posicionamento

# criando o gerador de imagem para data augmentation
datagen = ImageDataGenerator(
    shear_range = 0.4,
    zoom_range = 0.4,
    rotation_range = 25,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    horizontal_flip = True
)

# carrega uma imagem e transforma
img = load_img(EXAMPLE_PIC)
x = img_to_array(img) / 255.
x = x.reshape((1,) + x.shape)

# mostra 10 imagens alteradas da imagem carregada
plt.figure(figsize = (20,10))
plt.suptitle('Deformação das imagens para aprendizado', fontsize=28)

i = 0

for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3,5, i + 1)
    plt.grid(False)
    plt.imshow(batch.reshape(218,178,3))
    
    if i == 9:
        break
    i += 1
    
plt.show()
# Criação de aleatoriedade
# Importação da numpy
import numpy as np
# Data Train

x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)

# Data Train - Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
    x_train, y_train,
    batch_size = BATCH_SIZE,
)
# Validação
x_valid, y_valid = generate_df(1,'Male',VALIDATION_SAMPLES)

'''
# Validation - Data Preparation - Data Augmentation with generators
valid_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
)

valid_datagen.fit(x_valid)

validation_generator = valid_datagen.flow(
x_valid, y_valid,
)
'''

# Modelo CNN
# NOVAS IMPORTAÇÕES

#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import multi_gpu_model
# from keras.applications.vgg16 import VGG16
#from keras.models import Sequential
#from keras.layers import Dense


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
from tensorflow.keras import backend as K
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras import optimizers
import os
from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (5, 5), strides = (2,2), activation='relu')(i)
x = Conv2D(64, (3, 3), strides = (2,2), activation='relu')(x)
x = Conv2D(128, (3, 3), strides = (2,2), activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation = "softmax")(x)

model = Model(i, x)


'''# Import InceptionV3 Model
inc_model = InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

print("number of layers:", len(inc_model.layers))
#inc_model.summary()'''
# ENTROPIA: Execução do modelo por entropia
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
# Treino
r = model.fit(
    x_train,
    y_train,
    validation_data = (x_valid,y_valid),
    steps_per_epoch = int(TRAINING_SAMPLES / BATCH_SIZE),
    epochs = NUM_EPOCHS
)
