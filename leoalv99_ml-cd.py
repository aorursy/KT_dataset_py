import numpy as np
import pandas as pd
import zipfile
import os
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from keras.models import model_from_json

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip", "r") as z:
    z.extractall(".")

with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip", "r") as z:
    z.extractall(".")

# Lista de las categorias
categorias = []
# Lista de archivos dentro de los datos de train
archivos = os.listdir("train")
# Bucle para obtener los nombres de los archivos (Etiquetas de salida)
for archivo in archivos:
    categoria = archivo.split('.')[0]
    if categoria == 'dog':
        categorias.append("dog")
    else:
        categorias.append("cat")
        
# Se crea un dataframe en base del nombre del archivo y la categoria (tipo de animal)
df = pd.DataFrame({
    'archivo': archivos,
    'categoria': categorias
})
print(categoria, '\n', df)
# Separamos dos conjuntos, train(80% del dataset) y test(20%).
train, test = train_test_split(df, test_size=0.20, random_state=42)

# Reseteamos los index para vayan al comienzo de daatframe.
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Imprimimos el tamano de los dataframes.
print(train.shape, '<-->', test.shape)
# Generador De Entrenamiento (Preprocesamiento de las imagenes)
TAM_IMG = (128, 128)
"""
Generamos nuevas imagenes en base a las pre existentes, por ejmplo, podemos cambiar el rango de rotacion, 
cortamos la imagen, hacemos zoom, volteamos, entre otras tecnicas para que de esta manera podamos tener
una mayor cantidad de imagenes para entrenar y hacer el testing.

"""
train_image_data = ImageDataGenerator(
    rotation_range = 15,
    rescale = 1./255,
    shear_range = 0.1,
    zoom_range=0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)

# Procedemos ha generar las imagenes y preprocesarlas.
train_generador = train_image_data.flow_from_dataframe(
    train,
    'train/',
    x_col = 'archivo',
    y_col = 'categoria',
    target_size = TAM_IMG,
    class_mode = 'categorical',
    batch_size = 32
)

# Escalamos la imagen diviendo cada pixel para 255.
test_image_data = ImageDataGenerator(rescale=1./255)
test_generador = test_image_data.flow_from_dataframe(
    test,
    'train/',
    x_col = 'archivo',
    y_col = 'categoria',
    target_size = TAM_IMG,
    class_mode = 'categorical',
    batch_size = 32
)
# Modelo para el entrenamiento
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

modelo = Sequential()

modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.25))

modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.25))

modelo.add(Conv2D(128, (3, 3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.25))

modelo.add(Conv2D(128, (3, 3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Dropout(0.25))

modelo.add(Flatten())
modelo.add(Dense(512, activation = 'relu'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.5))
modelo.add(Dense(2, activation='softmax'))

modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
modelo.summary()
train_shape = train.shape[0]
test_shape = test.shape[0]

history = modelo.fit_generator(
    train_generador,
    epochs = 25,
    validation_data = test_generador,
    validation_steps = test_shape//64,
    steps_per_epoch = train_shape//64,
)
accuracy_training = history.history['accuracy']
accuracy_testing = history.history['val_accuracy']
epochs = 26
plt.rcParams["figure.figsize"] = (12,8)
plt.grid()
plt.plot(accuracy_training, color='b', label="Training accuracy")
plt.xticks(np.arange(1, epochs, 1))
plt.xlabel('EPOCHS')
plt.ylabel('Accuracy en Training')
plt.tight_layout()
plt.show()

plt.rcParams["figure.figsize"] = (12,8)
plt.grid()
plt.plot(accuracy_testing, color='r')
plt.xticks(np.arange(1, epochs, 1))
plt.xlabel('EPOCHS')
plt.ylabel('Accuracy en Testing')
plt.tight_layout()
plt.show()

def guardar_pesos(nombre_archivo='pesos.h5'):
    modelo.save_weights(nombre_archivo)

def guardar_modelo(nombre_archivo='modelo.json'):
    modelo_json = modelo.to_json()
    with open(nombre_archivo, "w") as json_file:
        json_file.write(modelo_json)

def cargar_modelo(archivo_modelo='modelo.json', archivo_pesos='pesos.h5'):
    with open(archivo_modelo, 'r') as f:
        modelo = model_from_json(f.read())
    
    modelo.load_weights(archivo_pesos)
    
    return modelo

modelo = cargar_modelo(archivo_modelo='../input/cats-v-dogs-leo/modelo.json', archivo_pesos='../input/cats-v-dogs-leo/pesos.h5')
archivos_test = os.listdir("test1")
test1 = pd.DataFrame({
    'archivo': archivos_test
})
muestras = test1.shape[0]
print(muestras)
# Preprocesamiento y generador de imagenes para testing.
TAM_IMG = (128, 128)

test_generador = ImageDataGenerator(rescale=1./255)
test_gen = test_generador.flow_from_dataframe(
    test1,
    'test1',
    x_col = 'archivo',
    y_col = None,
    class_mode = None,
    target_size = TAM_IMG,
    batch_size = 32,
    shuffle = False
)
prediccion = modelo.predict_generator(test_gen, steps=np.ceil(muestras/32))
test1['categoria'] = np.argmax(prediccion, axis = 1)
label_map = dict((v, k) for k, v in train_generador.class_indices.items())
test1['categoria'] = test1['categoria'].replace(label_map)
from io import BytesIO
from six.moves import urllib
from keras.preprocessing import image

TAM_IMG = (128, 128)


def predecir_img(URL):
    img = None
    with urllib.request.urlopen(URL) as url:
        img = load_img(BytesIO(url.read()), target_size=(128, 128))
    imagen = img
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    resultado = modelo.predict(img)
    val0 = resultado.tolist()[0][0]
    val1 = resultado.tolist()[0][1]
    if val0 > val1:
        return ('GATO', val0, imagen)
    elif val1 > val0:
        return ('PERRO', val1, imagen)

plt.figure(figsize=(12, 12))
    
animal, accuracy, imagen = predecir_img('https://static.iris.net.co/semana/upload/images/2019/6/18/620159_1.jpg')
print(animal +', probabilidad: ', str(round(accuracy*100, 2))+'%')
plt.subplot(5, 3, 1)
plt.imshow(imagen)

animal, accuracy, imagen = predecir_img('https://www.petdarling.com/articulos/wp-content/uploads/2014/08/gatos-persa.jpg?width=1200&enable=upscale')
print(animal +', probabilidad: ', str(round(accuracy*100, 2))+'%')
plt.subplot(5, 3, 2)
plt.imshow(imagen)

plt.tight_layout()
plt.show()
# METODO 
from keras.preprocessing import image

def display_stats(sample_id = 5):
    train_len = len([name for name in os.listdir('test1')])
    
    if sample_id < train_len:
        names = []
        datos = dict()
        for name in os.listdir('test1'):
            names.append(name)
        file = load_img('test1/'+ names[sample_id], target_size=(128, 128))
        datos['Imagen:'] = names[sample_id]
        img = image.img_to_array(file)
        datos['Valor minimo:'] = np.min(img)
        datos['Valor maximo:'] = np.max(img)
        datos['Shape:'] = img.shape
        img = np.expand_dims(img, axis=0)
        img /= 255
        
        resultado = modelo.predict(img)
        val0 = resultado.tolist()[0][0]
        val1 = resultado.tolist()[0][1]
        if val0 > val1:
            #return ('GATO', val0, imagen)
            datos['% Prediccion:'] = round(val0*100, 2)
            datos['ETIQUETA'] = 'GATO'
        elif val1 > val0:
            datos['% Prediccion:'] = round(val1*100, 2)
            datos['ETIQUETA'] = 'PERRO'
        for x, y in datos.items():
            print(x, '\t', y)
        print()
        plt.imshow(file)
 
# EJEMPLO PERRO
display_stats(27)
# EJEMPLO GATO
#display_stats(98)
