import numpy as np # biblioteca para algebra lineal
import pandas as pd # librería para procesamiento de data

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df_train.head()
df_train_x = df_train.iloc[:,1:] 
df_train_y = df_train.iloc[:,:1] 
ax = plt.subplots(1,5)
for i in range(0,5):   
    ax[1][i].imshow(df_train_x.values[i].reshape(28,28), cmap='gray')
    ax[1][i].set_title(df_train_y.values[i])
def cnn_model(result_class_size):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu')) #capa de convolución
    model.add(MaxPooling2D(pool_size=(2, 2))) #capa de pooling
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Dropout(0.2)) #20% de la capa cambia a cero(porcentaje que abandona)
    model.add(Flatten()) #convierte la matriz en un vector
    model.add(Dense(130, activation='relu')) #capa totalmente conectada
    model.add(Dense(50, activation='relu'))
    model.add(Dense(result_class_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model
arr_train_y = np_utils.to_categorical(df_train_y['label'].values)
model = cnn_model(arr_train_y.shape[1])
model.summary()

#normalización de tonos de gris a 1 ó 0
df_test = df_test / 255
df_train_x = df_train_x / 255
arr_train_x_28x28 = np.reshape(df_train_x.values, (df_train_x.values.shape[0], 28, 28, 1))
arr_test_x_28x28 = np.reshape(df_test.values, (df_test.values.shape[0], 28, 28, 1))
random_seed = 3
#analizamos el 80% del set de entrenamiento
split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(arr_train_x_28x28, arr_train_y, test_size = 0.08, random_state=random_seed)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                              factor=0.5,
                              patience=3, 
                              min_lr=0.00001)
# ImageDataGenerator permite generar variaciones en la imagen analizada
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1  
        )

datagen.fit(split_train_x)
model.fit_generator(datagen.flow(split_train_x,split_train_y, batch_size=64),
                              epochs = 30, validation_data = (split_val_x,split_val_y),
                              verbose = 2, steps_per_epoch=700 
                              , callbacks=[reduce_lr])
prediction = model.predict_classes(arr_test_x_28x28, verbose=0)
data_to_submit = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)), "Label": prediction})
data_to_submit.to_csv("result.csv", header=True, index = False)
from random import randrange
#escogemos 10 imagenes del set de entrenamiento
start_idx = randrange(df_test.shape[0]-10) 
fig, ax = plt.subplots(2,5, figsize=(15,8))
for j in range(0,2): 
  for i in range(0,5):
     ax[j][i].imshow(df_test.values[start_idx].reshape(28,28), cmap='gray')
     ax[j][i].set_title("Index:{} \nPrediction:{}".format(start_idx, prediction[start_idx]))
     start_idx +=1