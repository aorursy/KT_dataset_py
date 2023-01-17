# Inspiración:
#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-santander-value
#https://www.kaggle.com/mortido/keras-simple-model
# Librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Muestra archivos disponibles
import os
print(os.listdir("../input"))
# Carga bases disponibles
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# Identifica campos con valores no constantes
not_constant_attr = (train_df.nunique() != 1)
not_constant_attr_names = not_constant_attr[list(not_constant_attr)].index
#train_df[not_constant_attr_names].head()
train_df[not_constant_attr_names].shape
# Prepara bases
x_train = train_df[not_constant_attr_names].drop(["ID", "target"], axis=1)
x_test = test_df[not_constant_attr_names[2:]]

# Ajusta distribución de variables
x_train = np.log1p(x_train)
x_test = np.log1p(x_test)

# Junta bases para obtener estadísticos de media y desviación estándar
x_total = pd.concat((x_test, x_train), axis=0).replace(0,  np.nan)

# Escala valores
x_train = (x_train - x_total.mean()) / x_total.std()
x_test = (x_test - x_total.mean()) / x_total.std()

# Escala variable objetvo
y_train = np.log1p(train_df["target"].values)
# Genera base de entrenamiento y validación
dev_x, val_x, dev_y, val_y = train_test_split(x_train, y_train, test_size = 0.2, random_state = 777)
# Definición del modelo
from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(0.05), activation='linear', input_shape=(x_train.shape[1],)))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.05), activation='linear'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.05), activation='linear'))
model.add(layers.LeakyReLU(alpha=0.3))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
import keras.backend as K
from keras.optimizers import Adam

# Función de pérdida-metrica
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
# Compilando el modelo
model.compile(optimizer=Adam(lr=0.0001),
              loss=root_mean_squared_error,
              metrics=[root_mean_squared_error])
import keras
batch_size = 128
epochs = 100

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=5, verbose=0, mode='auto')

# Entrenando el modelo
history = model.fit(#dev_x,
                    #dev_y,
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size)#,
                    #callbacks=[lr_scheduler, es],
                    #validation_data=(val_x, val_y))
history_dict = history.history
#history_dict.keys()
#history_dict
# Muestra el gráfico del entrenamiento
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 150 + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
rmse_values = history_dict['root_mean_squared_error']
val_rmse_values = history_dict['val_root_mean_squared_error']
plt.plot(epochs, rmse_values, 'bo', label='Training rmse')
plt.plot(epochs, val_rmse_values, 'b', label='Validation rmse')
plt.title('Training and validation rmse')
plt.xlabel('Epochs')
plt.ylabel('Rmse')
plt.legend()
plt.show()
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#plt.figure(figsize=(12,8))
#sns.distplot( np.log1p(train_df["target"].values), bins=1413, kde=False)
#plt.xlabel('Target', fontsize=12)
#plt.title("Log of Target Histogram", fontsize=14)
#plt.show()
model.predict(x_test.iloc[0:3])
model.summary()
pred_keras = np.expm1(model.predict(x_test))
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_keras

print(sub.head())
sub.to_csv('keras_modelo_v3.csv', index=False)
#import gc
#gc.collect()