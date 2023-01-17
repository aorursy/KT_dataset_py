import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/training/training.csv')
df.dropna(inplace=True)
df.shape
from joblib import Parallel, delayed

def format_img(x):
    return np.asarray([int(e) for e in x.split(' ')], dtype=np.uint8).reshape(96,96)

with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    x = ex(delayed(format_img)(e) for e in df.Image)
    
x = np.stack(x)[..., None]
x.shape
y = df.iloc[:, :-1].values
y.shape
def show(x, y=None):
    plt.imshow(x[..., 0], 'gray')
    if y is not None:
        points = np.vstack(np.split(y, 15)).T
        plt.plot(points[0], points[1], 'o', color='red')
        
    plt.axis('off')

sample_idx = np.random.choice(len(x))    
show(x[sample_idx], y[sample_idx])
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_val.shape
# Observamos las dimensiones de "y"
y_train.shape, y_val.shape
# Normalizar las imágenes (1pt)

# Opto por normalizar utilizando la media y las desviación estándar
# Estiramos las imágenes:
x_train_norm=x_train[:,:,:,:]
x_val_norm=x_val[:,:,:,:]
x_train_norm=x_train_norm.reshape([1712,96*96,1])
x_val_norm=x_val_norm.reshape([428,96*96,1])

# Normalización:
mu=x_train_norm.mean()
sigma=x_train_norm.std()

x_train_norm=(x_train_norm - mu)/sigma
x_val_norm=(x_val_norm - mu)/sigma # Se normaliza siempre con el mu y sigma de los datos de entrenamiento

x_train_norm.shape, x_val_norm.shape, x_train_norm.mean(), x_train_norm.std(), x_val_norm.mean(), x_val_norm.std()


# Observamos arriba que efectivamente están normalizados. Ahora retornamos a las dimensiones de matriz de imagen:
x_train_norm=x_train_norm.reshape([1712,96,96,1])
x_val_norm=x_val_norm.reshape([428,96,96,1])

x_train_norm.shape, x_val_norm.shape
# Observamos una imagen para ver si está todo en orden:
show(x_train_norm[15], y_train[15])
# Definir correctamente la red neuronal (5 pts)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import regularizers

model=Sequential([
    Conv2D(72,4,input_shape=(96,96,1),activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)),
    AvgPool2D(pool_size=(2,2)),
    Conv2D(48,2,activation='relu',use_bias=False,kernel_initializer='he_normal' ,kernel_regularizer=regularizers.l2(0.01)), #Según clase, no se debe inicializar bias antes de un batchnorm
    BatchNormalization(),
    Flatten(),
    Dropout(0.5), #Actúa como regularizador
    Dense(48,activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)), #Importante utilizar he initialization para relu
#     Dropout(0.2), #Actúa como regularizador
#     Dense(40,activation='relu', kernel_initializer='he_normal'), #Importante utilizar he initialization para relu
    Dense(30, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01)) # No hay activación acá por ser un problema de regresión
])

model.compile(optimizer=Adam(0.01),loss='mse',metrics=['mae']) # Settings según indicaciones
# Resumen de las capas del modelo:
model.summary(), model.input, model.output
# Entrenar la red neuronal (2 pts)
log=model.fit(x_train_norm, y_train, epochs=150, batch_size=256, validation_data=[x_val_norm,y_val])
# Resultado del entrenamiento
# - mae entre 10 y 15 (3 pts)
# - mae entre 8 y 11 (5 pts)
# - mae entre 5 y 8 (7 pts)
# - mae menor o igual a 4.0 (9 pts)

print(f'MAE final: {model.evaluate(x_val, y_val)[1]}')
print(f'MAE final: {model.evaluate((x_val-mu)/sigma, y_val)[1]}')
print(f'MAE final: {model.evaluate(x_val_norm, y_val)[1]}')
# Ver la perdida en el entrenamiento
def show_results(*logs):
    trn_loss, val_loss, trn_acc, val_acc = [], [], [], []
    
    for log in logs:
        trn_loss += log.history['loss']
        val_loss += log.history['val_loss']
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(trn_loss, label='train')
    ax.plot(val_loss, label='validation')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.legend()
    
show_results(log)
# Función para visualizar un resultado
def show_pred(x, y_real, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    for ax in axes:
        ax.imshow(x[0, ..., 0], 'gray')
        ax.axis('off')
        
    points_real = np.vstack(np.split(y_real[0], 15)).T
    points_pred = np.vstack(np.split(y_pred[0], 15)).T
    axes[0].plot(points_pred[0], points_pred[1], 'o', color='red')
    axes[0].set_title('Predictions', size=16)
    axes[1].plot(points_real[0], points_real[1], 'o', color='green')
    axes[1].plot(points_pred[0], points_pred[1], 'o', color='red', alpha=0.5)
    axes[1].set_title('Real', size=16)
# Ordenamos el set de validación según el ranking de errores obtenidos al predecir el set de validación:
predicciones_val=model.predict(x_val_norm)
residuales_val=np.abs(predicciones_val - y_val)
mad_val=np.sum(residuales_val, axis=1)/30

indices=mad_val.argsort()
indices.shape
# Mostrar 5 resultados aleatorios del set de validación (1 pt)
for _ in range(5):
    index = np.random.choice(x_val_norm.shape[0])
    sample_x = x_val_norm[index, None]
    sample_y = y_val[index, None]
    pred = model.predict(sample_x)
    show_pred(sample_x, sample_y, pred)
# Mostrar las 5 mejores predicciones del set de validación (1 pt)
for i in range(5):
    sample_x = x_val_norm[indices[i], None]
    sample_y = y_val[indices[i], None]
    pred = model.predict(sample_x)
    show_pred(sample_x, sample_y, pred)
# Mostrar las 5 peores predicciones del set de validación (1 pt)
for i in [-1,-2,-3,-4,-5]:
    sample_x = x_val_norm[indices[i], None]
    sample_y = y_val[indices[i], None]
    pred = model.predict(sample_x)
    show_pred(sample_x, sample_y, pred)