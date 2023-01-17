# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf



from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation

from keras.models import Model

from keras.optimizers import Adam

from keras.initializers import glorot_normal



from sklearn import preprocessing
# Definición de la arquitectura de la red

def crear_clasificador():

    x_input = Input(shape=(8,))

    x = Dense(500, activation='relu', kernel_initializer=glorot_normal())(x_input)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(1000, activation='relu', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(1200, activation='relu', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(1500, activation='relu', kernel_initializer=glorot_normal())(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    y = Dense(1, activation='sigmoid', kernel_initializer=glorot_normal())(x)

    

    model = Model(x_input, y)

    model.summary()

    

    return model
# Creación del modelo a entrenar

clasificador = crear_clasificador()

clasificador.compile(loss='binary_crossentropy', optimizer=Adam(0.001, 0.5), metrics=['binary_accuracy'])
# Carga de datos y eliminación de columnas innecesarias

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data_X = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data_Y = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



train_data.drop('Cabin', axis=1, inplace=True)

test_data_X.drop('Cabin', axis=1, inplace=True)

train_data.drop('Ticket', axis=1, inplace=True)

test_data_X.drop('Ticket', axis=1, inplace=True)

train_data.drop('PassengerId', axis=1, inplace=True)

test_data_X.drop('PassengerId', axis=1, inplace=True)







train_data.head()
# Extracción del título del valor en "Name" y creación de la columna "Title"

total_data = [train_data, test_data_X]



for row in total_data:

    row['Title'] = row['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



# Mapping de los títulos a números

    mapeo = {"Miss": 0, "Mr": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for row in total_data:

    row['Title'] = row['Title'].map(mapeo)    



# Eliminación de la columna "Name"

train_data.drop('Name', axis=1, inplace=True)

test_data_X.drop('Name', axis=1, inplace=True)



train_data.head()
# Relleno de celdas en la columna "Embarked" sin valor, asignándoles el que más se repite

for row in total_data:

    row['Embarked'].fillna(row['Embarked'].value_counts().idxmax(), inplace=True)





mapeo = {'S' : 0, 'C' : 1, 'Q': 2}



for row in total_data:

    row['Embarked'] = row['Embarked'].map(mapeo)

    

train_data.head()
# Relleno de celdas en la columna "Embarked" sin valor, asignándoles la media de todos los valores 

for row in total_data:

    row["Fare"].fillna(row["Fare"].median(skipna=True), inplace=True)

    row['Fare'] = row['Fare'].astype(int)

train_data.head()
# Relleno de celdas en la columna "Age" sin valor, asignándoles la media de todos los valores  

for row in total_data:

    row["Age"].fillna(row["Age"].median(skipna=True), inplace=True)

    row['Age'] = row['Age'].astype(int)

train_data.head()
# Mapping de la columna "Sex" a números

mapeo = {'male': 0, 'female': 1}

for row in total_data:

    row['Sex'] = row['Sex'].map(mapeo)

train_data.head()
# Extracción de labels de entrenamiento (columna "Survived") 

train_data_Y = train_data['Survived']

train_data_X = train_data.drop('Survived', axis=1)



train_data_X.head()
# Normalización de datos de entrada

train_data_X = preprocessing.normalize(train_data_X)

test_data_X = preprocessing.normalize(test_data_X)



# Entrenamiento del modelo

clasificador.fit(train_data_X, train_data_Y, epochs=70)



# Evaluación de accuracy con set de testeo

test = clasificador.evaluate(test_data_X, test_data_Y['Survived'])



print(test)
predictions = clasificador.predict(test_data_X)

submission = test_data_Y[['PassengerId']].copy()

submission['Survived'] = np.round(predictions).astype(int)

submission.head()

# Pasar DataFrame de Pandas a archivo CSV

submission.to_csv('submission.csv', index=False)
# Descargar CSV

from IPython.display import FileLink

FileLink(r'submission.csv')