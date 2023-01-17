# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/bankfull/bank-full.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Codificar las variables categóricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Paso 1 - labelencoder para convertir categorías a números
labelencoder_X_servicio = LabelEncoder()
X[:, 1] = labelencoder_X_servicio.fit_transform(X[:, 1])

labelencoder_X_estado_civil = LabelEncoder()
X[:, 2] = labelencoder_X_estado_civil.fit_transform(X[:, 2])

labelencoder_X_estudios = LabelEncoder()
X[:, 3] = labelencoder_X_estudios.fit_transform(X[:, 3])

labelencoder_X_contacto = LabelEncoder()
X[:, 8] = labelencoder_X_contacto.fit_transform(X[:, 8])

labelencoder_X_mes = LabelEncoder()
X[:, 10] = labelencoder_X_mes.fit_transform(X[:, 10])

labelencoder_X_observacion = LabelEncoder()
X[:, 15] = labelencoder_X_observacion.fit_transform(X[:, 15])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",       
         OneHotEncoder(), 
         [1,2,3,8,10,15] #,2,3,8,10,15             
         )
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X.tolist())
X = X[:, [1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,20,21,23,24,25,26,27,28,29,30,31,32,33,35,36,37]]
# Dividir el dataset en train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Escalamos las características
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
redNeuronalClasificacion = Sequential()
redNeuronalClasificacion.add(Dense(input_dim=32, units=100,kernel_initializer='uniform', activation='relu'))
redNeuronalClasificacion.add(Dropout(rate=0.7))
#redNeuronalClasificacion.add(Dense(units=50,kernel_initializer='uniform', activation='relu'))
#redNeuronalClasificacion.add(Dropout(rate=0.1))
redNeuronalClasificacion.add(Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))
redNeuronalClasificacion.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
OBSERVACIONES_POR_LOTE = 2

# Pasadas completas por todo el dataset.
EPOCHS = 150

history = redNeuronalClasificacion.fit(
    X_train, 
    y_train, 
    batch_size=OBSERVACIONES_POR_LOTE, 
    epochs= EPOCHS,
    validation_data=(X_test, y_test)
)
# Plot Precisión

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plot Pérdida

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()