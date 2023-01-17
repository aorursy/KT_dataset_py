import numpy as np
import pandas as pd
import tensorflow as tf

import keras.models
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('../input/airpressure/Folds5x2_pp.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.05, random_state = 0)
ann = Sequential()
ann.add(Dense(units=6, activation='softmax'))
ann.add(Dense(units=6, activation='softmax'))
ann.add(Dense(units=1))
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
my_callbacks = [
    EarlyStopping(patience=10),
]
bitacora = ann.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val), verbose=2, callbacks=my_callbacks)

#ann.fit(X_train, y_train, batch_size = 32, epochs = 70)
ann.summary()
import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

axs.plot(bitacora.history['loss'], label='loss')
axs.plot(bitacora.history['val_loss'], label='val_loss')

plt.legend();
plt.show();
ann.save('MLPModel.h5') #Guardando el modelo para futuras pruebas
len(y_val)
y_pred = ann.predict(X_val)
np.set_printoptions(precision=2) #Para mostrar solo 2 decimales
from sklearn.metrics import r2_score
r2_score(y_val, y_pred)
from sklearn.model_selection import GridSearchCV
#from tf.keras.wrappers.scikit_learn import KerasClassifier
# Funcion que debe devolver nuestra red neuronal. Aqui se le llenan con parametros dinamicos segun lo que se necesite
def create_model(optimizer, init_mode, activation, neurons):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=neurons,  kernel_initializer=init_mode, activation=activation))
    ann.add(tf.keras.layers.Dense(units=neurons,  kernel_initializer=init_mode, activation=activation))
    ann.add(tf.keras.layers.Dense(units=1))
    ann.compile(optimizer = optimizer, loss = 'mean_squared_error') #Aca puede ir metrics = ['accuracy'] para clasificacion
    return ann
# definir la busqueda de parametros
batch_size = [32] #[8,16,32,64,128]
epochs = [70,100] #[10,25,50,100,150,250]
optimizer = ['adam','SGD']#['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['glorot_uniform']#['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax','relu']#['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
neurons = [5,6]
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, init_mode=init_mode, activation = activation, neurons = neurons)
# La siguiente linea funciona solamente para modelos de clasificacion
#modelo =  tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)

#Para modelos de regresion (mi caso)
modelo =  tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)
grid = GridSearchCV(estimator=modelo, param_grid=param_grid, n_jobs=-1, cv=3)# Quitar scoring para problemas de clasificacion
grid_result = grid.fit(X_train, y_train)
# Resumen de resultados
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
modeloFinal=create_model(activation='relu', init_mode='glorot_uniform', neurons=6, optimizer = 'adam')
bitacoraMejor = modeloFinal.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data=(X_val, y_val), verbose=2)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

axs.plot(bitacoraMejor.history['loss'], label='loss')
axs.plot(bitacoraMejor.history['val_loss'], label='val_loss')

plt.legend();
plt.show();
#Prediccion final
y_pred2 = modeloFinal.predict(X_test)
#Evaluacion del modelo
from sklearn.metrics import r2_score
r2_score(y_test, y_pred2)
modeloFinal.summary()
modeloFinal.save('MejorMLPModel.h5') #Guardando el modelo para futuras pruebas