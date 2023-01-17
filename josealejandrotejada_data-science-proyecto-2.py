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
import numpy as np # linear algebra

import pandas as pd # para analisis de CSV

import tensorflow as tf #tensor flow para imgenes

import datetime, os #datetime

import math #para calculos

import matplotlib.pyplot as plt #para graficar

import seaborn as sns #para lindas graficas

from sklearn.model_selection import train_test_split
#Cargamos los dataframes

train_df = pd.read_csv('/kaggle/input/rsna-bone-age/boneage-training-dataset.csv')

test_df = pd.read_csv('/kaggle/input/rsna-bone-age/boneage-test-dataset.csv')

print("Done")
train_df.head()
train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png') 

test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x)+'.png') 



train_df.head() 
train_df['gender'] = train_df['male'].apply(lambda x: 'masculino' if x else 'femenino')

print(train_df['gender'].value_counts())



#contamos la cantidad de valores que  hay y hacemos una gráfica

sns.countplot(x = train_df['gender'])
# Paciente mas joven del datasent

print('Edad mínima: ' + str(train_df['boneage'].min()) + ' meses')
#Paciente mas grande, en meses

print('Edad máxima: ' + str(train_df['boneage'].max()) + ' meses')
#Media de las edades oseas

mediaOseaGeneral = train_df['boneage'].mean()

print('Media osea: ' + str(mediaOseaGeneral))
#Desviación estándar

desviacionEdadesOseas = train_df['boneage'].std()

desviacionEdadesOseas
train_df['boneAgeZ'] = (train_df['boneage'] - mediaOseaGeneral)/(desviacionEdadesOseas)

print(train_df.head())
#Graficamos histogramas

train_df['boneage'].hist(color = 'aquamarine')

plt.xlabel('Edad osea en meses')

plt.ylabel('Cantidad de niños')

plt.title('Niños por grupo: masculino o femenino')
train_df['boneAgeZ'].hist(color = 'orange')

plt.xlabel('Punteo de Edad osea con ajuste Z')

plt.ylabel('Conteo de niños')

plt.title('Relación entre el valor Z y el número de niños.')
male = train_df[train_df['gender'] == 'masculino']

female = train_df[train_df['gender'] == 'femenino']

fig, ax = plt.subplots(2,1)

ax[0].hist(male['boneage'], color = 'aquamarine')

ax[0].set_ylabel('Cantidad de niños')

ax[1].hist(female['boneage'], color = 'violet')

ax[1].set_xlabel('Edad en meses')

ax[1].set_ylabel('Número de niñas')

fig.set_size_inches((10,7))
#Relación entre la edad osea y el género

sns.swarmplot(x = train_df['gender'], y = train_df['boneage'])
#Separamos la data

df_train, df_valid = train_test_split(train_df, test_size = 0.2, random_state = 0)



print(df_train.head())
print(df_valid.head())
import matplotlib.image as mpimg



#agarramos 5 valores

for filename, boneage, gender in train_df[['id','boneage','gender']].sample(6).values:

    #le pasamos el path

    img = mpimg.imread('/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'+ filename)

    plt.imshow(img)#plotteamos la imagen

    #dividimos los meses en 12 para dar el valor de la edad

    plt.title('Nombre imagen:{}  Edad osea: {} años  Género: {} Dimensiones: h,w{}'.format(filename, boneage/12, gender, img.shape))

    plt.axis('off')

    plt.show()
print(df_train) #10088 datos

print(train_df) #12611 datos
import matplotlib.image as mpimg

listDim = []

#agarramos 10000 valores

for filename, boneage, gender in train_df[['id','boneage','gender']].sample(10000,random_state=123).values:

    #le pasamos el path

    img = mpimg.imread('/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'+ filename)

    dim = img.shape

    listDim.append(dim)

    

df_IDimensiones = pd.DataFrame(listDim,columns=['i.height','i.width'])

print('pos(0,0): '+str(listDim[0][0]) + ', Len: '+str(len(listDim)))

print('__________________________')

print(df_IDimensiones.head())
df_IDimensiones['i.height'].hist(color = 'orange')

plt.xlabel('Altura de la imagen')

plt.ylabel('Imagenes')

plt.title('Distribucion de la altura de las imagenes')
df_IDimensiones['i.width'].hist(color = 'blue')

plt.xlabel('Ancho de la imagen')

plt.ylabel('Imagenes')

plt.title('Distribucion del ancho de las imagenes')
print(max(listDim))
print(min(listDim))
df_IDimensiones['i.height'].count()
df_IDimensiones['i.height'].mean()
df_IDimensiones['i.width'].mean()
df_IDimensiones['i.width'].std()
df_IDimensiones['i.height'].std()
#importamos librerias

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from  keras.applications.xception import preprocess_input 
#Colocamos el tamaño de la imagen que queremos

img_size = 256



train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator = train_data_generator.flow_from_dataframe(

    dataframe = df_train,

    directory = '/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset',

    x_col= 'id',#nombre de la columa x

    y_col= 'boneAgeZ',#Nombre de la columna Y

    batch_size = 32,#batch size

    seed = 25, #semilla de 25 para que sea reproducible

    shuffle = True,#random agarrar

    class_mode= 'other',#el tipo de clase de data, como no es normal, usamos other por ser imagen

    flip_vertical = True,#se hace flip

    color_mode = 'rgb',# las colocamos RGB

    target_size = (img_size, img_size))#colocamos el tamaño de 256x256 pixeles



print("Data de train lograda")
train_generator
#Data de validación

val_generator = val_data_generator.flow_from_dataframe(

    dataframe = df_valid,

    directory = '/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset',

    x_col = 'id',

    y_col = 'boneAgeZ',

    batch_size = 32,

    seed = 42,

    shuffle = True,

    class_mode = 'other',

    flip_vertical = True,

    color_mode = 'rgb',

    target_size = (img_size, img_size))



#Data de test

test_data_generator = ImageDataGenerator(

    preprocessing_function = preprocess_input,

    rescale = 1.0/255

)



test_generator = test_data_generator.flow_from_directory(

    directory = '/kaggle/input/rsna-bone-age/boneage-test-dataset',

    shuffle = True,

    class_mode = None,

    color_mode = 'rgb',

    target_size = (img_size,img_size))
test_X, test_Y = next(val_data_generator.flow_from_dataframe( 

                            df_valid, 

                            directory = '/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset',

                            x_col = 'id',

                            y_col = 'boneAgeZ', 

                            target_size = (img_size, img_size),

                            batch_size = 2523,

                            class_mode = 'other'

                            )) 



print("Método 2 hecho")