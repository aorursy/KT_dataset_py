# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time #para poder sacar el tiempo de ejecucion



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importa la libreria tensorflow

import tensorflow as tf 



#Importa información para el modelo

from keras.models import Model 



#Importa información para capas

from keras.layers import Input, Dense, Dropout,BatchNormalization, Activation 



#Importa los optimizadores

from keras.optimizers import Adam 



#Importa inicializacion para los pesos

from keras.initializers import glorot_normal 



#Importa para normalizar los datos de entrada

from sklearn import preprocessing

#Arquitectura de la red

def c_clasificador():

    #Recibe un vector de entrada de cualquier tamaño con 8 columnas, ya que son las que quedan luego de los drop

    x_input = Input(shape=(8,))

    

    #Capa de neuronas conectadas, es decir, la primera capa oculta

    x = Dense(650, activation='relu', kernel_initializer=glorot_normal())(x_input)

    

    #Batch Normalization

    x = BatchNormalization()(x)

    

    #Probabilidad de que se elimine una neurona

    x = Dropout(0.3)(x)

    

    #Capa de neuronas conectadas, es decir, la segunda capa oculta

    x = Dense(600, activation='relu', kernel_initializer=glorot_normal())(x)

    #Probabilidad de que se elimine una neurona

    x = Dropout(0.3)(x)

    

    #Capa de neuronas conectadas, es decir, la tercera capa oculta

    x = Dense(500, activation='relu', kernel_initializer=glorot_normal())(x)

    #Probabilidad de que se elimine una neurona

    x = Dropout(0.3)(x)

    

    #Capa de neuronas conectadas, es decir, la cuarta capa oculta

    x = Dense(300, activation='relu', kernel_initializer=glorot_normal())(x)

    #Probabilidad de que se elimine una neurona

    x = Dropout(0)(x)

    

    

    #Capa de salida

    y = Dense(1, activation='sigmoid',kernel_initializer=glorot_normal())(x)

    

    #Especifica cual es la entrada de entrada y cual es la de salida

    model = Model(x_input,y)

    

    #Imprime las especificaciones de la red neuronal

    model.summary()

    return model
#Creo el modelo

clasificador = c_clasificador()

clasificador.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['binary_accuracy'])

tiempo_inicio = time.time()
#Carga del dataset

datos_entrenamiento = pd.read_csv('/kaggle/input/titanic/train.csv')

datos_prueba = pd.read_csv('/kaggle/input/titanic/test.csv')

datos_totales = [datos_entrenamiento, datos_prueba]

submission = datos_prueba[['PassengerId']].copy()
#Extrae el titulo del nombre

for iteracion_titulo in datos_totales:

    iteracion_titulo['Title'] = iteracion_titulo['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



e_titulo = {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Sir': 4, 'Capt':4, 'Mme': 4, 

            'Dona': 4, 'Don': 4, 'Jonkheer': 4, 'Lady': 4, 'Ms': 4, 'Countess': 4,

            'Mlle': 4, 'Major': 4, 'Col': 4, 'Rev': 4, 'Dr': 4, 'Master': 4}

for iteracion_titulo in datos_totales:

    iteracion_titulo['Title'] = iteracion_titulo['Title'].map(e_titulo)



datos_prueba.head()

#Quita la columna name

datos_entrenamiento.drop('Name', axis=1, inplace=True)

datos_prueba.drop('Name', axis=1, inplace=True)



#Quita la columna cabina

datos_entrenamiento.drop('Cabin', axis=1, inplace=True)

datos_prueba.drop('Cabin', axis=1, inplace=True)



#Quita la columna ticket

datos_entrenamiento.drop('Ticket', axis=1, inplace=True)

datos_prueba.drop('Ticket', axis=1, inplace=True)



#Quita la columna ID

datos_entrenamiento.drop('PassengerId', axis=1, inplace=True)

datos_prueba.drop('PassengerId', axis=1, inplace=True)





datos_prueba.head()
#Extrae la entrada de embarcación

for iteracion_embarcacion in datos_totales:

    iteracion_embarcacion['Embarked'].fillna(iteracion_embarcacion['Embarked'].value_counts().idxmax(), inplace=True)

    

e_embarcacion = {'S': 1, 'Q': 2, 'C': 3}



for iteracion_embarcacion in datos_totales:

    iteracion_embarcacion['Embarked'] = iteracion_embarcacion['Embarked'].map(e_embarcacion)



datos_prueba.head()
#Busca la media de la columna fare y pone ese valor a los que no tengan nada

for iteracion_fare in datos_totales:

    iteracion_fare["Fare"].fillna(iteracion_fare["Fare"].median(skipna=True), inplace=True)

    iteracion_fare['Fare'] = iteracion_fare['Fare'].astype(int)
#Busca la madia de la columna edad y pone ese valor a los que no tengan nada

for iteracion_edad in datos_totales:

    iteracion_edad["Age"].fillna(iteracion_edad["Age"].median(skipna=True), inplace=True)

    iteracion_edad['Age'] = iteracion_edad['Age'].astype(int)
#Extrae el sexo de la persona

e_sexo = {'male': 0, 'female': 1}



for iteracion_sexo in datos_totales:

    iteracion_sexo['Sex'] = iteracion_sexo['Sex'].map(e_sexo)



datos_prueba.head()
#Extraer los que sobrevivieron para poder comparar

datos_entrenamiento_y = datos_entrenamiento['Survived']



#Deja de tomar en cuenta la columna survived

datos_entrenamiento_x = datos_entrenamiento.drop('Survived', axis = 1)



datos_prueba.head()
#Definir el algoritmo de entrenamiento

def entrenamiento(epochs,datos_entrenamientox, datos_entrenamientoy ):

    #Normalizar los datos para que se entrene mejor la red

    datos_entrenamiento_x = preprocessing.normalize(datos_entrenamientox)

    datos_prueba_x = preprocessing.normalize(datos_prueba)

    

    #Entrena la red comparando los datos de x con los datos de y en los epochs

    clasificador.fit(datos_entrenamientox,datos_entrenamientoy, epochs=epochs )
#Entrena por 250 epochs

entrenamiento(250, datos_entrenamiento_x, datos_entrenamiento_y)



#Evaluacion del entrenamiento

prueba = clasificador.predict(datos_prueba)

print(prueba)

#Tiempo de ejecución

print("Tiempo de ejecución %s segundos" % (time.time() - tiempo_inicio))
#Exportar data en .csv para comparar con los datos

y_salida = clasificador.predict(datos_prueba)

submission['Survived'] = np.rint(y_salida).astype(int)

print(submission)

submission.to_csv('submission.csv', index=False)
