import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/training/training.csv')
df.dropna(inplace=True)
df.shape
df_test=pd.read_csv('../input/test/test.csv')

from joblib import Parallel, delayed

def format_img(x):
    return np.asarray([int(e) for e in x.split(' ')], dtype=np.uint8).reshape(96,96)

with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    x = ex(delayed(format_img)(e) for e in df.Image)
with Parallel(n_jobs=10, verbose=1, prefer='threads') as ex:
    test = ex(delayed(format_img)(e) for e in df_test.Image)
test = np.stack(test)[..., None]
x = np.stack(x)[..., None]
x.shape, test.shape
plt.imshow(x[3,:,:,0])
y = df.iloc[:, :-1].values
y.shape
y[1,:]
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
x[:,95,95,0]
# Normalizar las imágenes (1pt)
# Se requiere que las imágenes estén en el rango de [0,1], solo se dividirá entre 255, la primera capa de la red será un batchnormalizer y normalizará
x_train=(x_train/255)
x_val=(x_val/255)
test=(test/255)
#Como primera iteración utilizaremos ResNet 50 con los entrenados con imagenes de imagenet
#Se preprocesará de acuerdo al preprocesamiento de resnet 
from keras.applications.resnet50 import ResNet50, preprocess_input

test.shape
test=np.array([test[:,:,:,0],test[:,:,:,0],test[:,:,:,0]])
test=np.swapaxes(test,0,1)
test=np.swapaxes(test,1,2)
test=np.swapaxes(test,2,3)
test.shape
x_train[:,:,:,0].shape
x_train=np.array([x_train[:,:,:,0],x_train[:,:,:,0],x_train[:,:,:,0]])
x_train.shape
x_val[:,:,:,0].shape
x_val=np.array([x_val[:,:,:,0],x_val[:,:,:,0],x_val[:,:,:,0]])
x_val.shape

x_train=np.swapaxes(x_train,0,1)
x_train=np.swapaxes(x_train,1,2)
x_train=np.swapaxes(x_train,2,3)
x_train.shape
x_val=np.swapaxes(x_val,0,1)
x_val=np.swapaxes(x_val,1,2)
x_val=np.swapaxes(x_val,2,3)
x_val.shape
#x_train[:,:,:,0] = preprocess_input(x_train[:,:,:,0])
#x_val[:,:,:,0] = preprocess_input(x_val[:,:,:,0])
# Definir correctamente la red neuronal (5 pts)
#Se utilizarán las capas de convoluciones con los pesos fijos y solo se entrenará la capa densa final

base_model = ResNet50(include_top=False, input_shape=(96,96,3), pooling='avg')
base_model.trainable = False
base_model.summary()
y.shape
from keras.models import Sequential 
from keras.layers import Dense, Flatten,BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import regularizers
#Se propondra la capa densa
top_model = Sequential([
    Dense(512, activation='relu', input_shape=(2048,),kernel_initializer='he_normal'), #la capa resnet50 termina con 2048 inputs en una sola dimensión
    Dense(256, activation='relu',kernel_initializer='he_normal'),
    Dropout(0.7),
    Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal'),
    Dropout(0.7),
    Dense(96, activation='relu',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal'),
    Dropout(0.7),
    Dense(48, activation='relu',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='he_normal'),
    Dense(30)
])
top_model.compile(loss='mse', optimizer=Adam(0.001), metrics=['mae'])
top_model.summary()
#El modelo final consta de la capa convolucional de resnet y la capa densa propia
final_model = Sequential([base_model, top_model])
final_model.compile(loss='mse', optimizer=Adam(0.001), metrics=['mae'])
final_model.summary()
# Entrenar la red neuronal (2 pts)
#Pre computamos los pesos de la capa convolucional
precomputed_train = base_model.predict(x_train, batch_size=256, verbose=1)
precomputed_train.shape
precomputed_val = base_model.predict(x_val, batch_size=256, verbose=1)
precomputed_val.shape
log = top_model.fit(precomputed_train, y_train, epochs=600, batch_size=256, validation_data=[precomputed_val, y_val])


# Resultado del entrenamiento
# - mae entre 10 y 15 (3 pts)
# - mae entre 8 y 11 (5 pts)
# - mae entre 5 y 8 (7 pts)
# - mae menor o igual a 4.0 (9 pts)

print(f'MAE final: {final_model.evaluate(x_val, y_val)[1]}')
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
x_val[0,None].shape
sample_x = x_train[0, None]
sample_y = y_val[0, None]
pred = final_model.predict(sample_x)
show_pred(sample_x, sample_y, pred)
results=final_model.predict(test)
results
lookup = pd.read_csv('../input/IdLookupTable.csv')

lookid_list = list(lookup['FeatureName'])
imageID = list(lookup['ImageId']-1)
pre_list = list(results)

rowid = lookup['RowId']
rowid=list(rowid)
len(rowid)

feature = []
for f in list(lookup['FeatureName']):
    feature.append(lookid_list.index(f))
    preded = []
for x,y in zip(imageID,feature):
    preded.append(results[x][y])
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)
submission.to_csv('submission_resnet.csv',index = False)
# Mostrar 5 resultados aleatorios del set de validación (1 pt)

# Mostrar las 5 mejores predicciones del set de validación (1 pt)

# Mostrar las 5 peores predicciones del set de validación (1 pt)
