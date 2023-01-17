# Este entorno de Python 3 es similar a Jupyter Notebook

# Viene con varias librerías instaladas. Para más információn podés consultar 

# la imagen de docker que utiliza (https://github.com/kaggle/docker-python)



import numpy as np 

import pandas as pd

import keras

import matplotlib.pyplot as plt

%matplotlib inline



# Los archivos del zip "chest_xray.zip" están disponibles automáticamente en la 

# carpeta "/kaggle/input/aa2019unlp/chest_xray"

# (o "../input/aa2019unlp/chest_xray")

# Por ejemplo, el siguiente código imprime todos los archivos disponibles:

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Cualquier resultado que guarden en el directorio actual queda como un "output" en el workspace

from keras.preprocessing.image import load_img

dataset_folderpath="/kaggle/input/aa2019unlp/chest_xray"

print("Hay 3 directorios con los tres subconjuntos de datos: ",os.listdir(dataset_folderpath))

print("La carpeta de cada subconjunto tiene dos subcarpetas: NORMAL y PNEUMONIA")



train_folderpath = os.path.join(dataset_folderpath,"train")

val_folderpath = os.path.join(dataset_folderpath,"val")

test_folderpath = os.path.join(dataset_folderpath,"test")



img_name = '1500_normal.jpg'

image_path= f'normal/{img_name}'

img_normal = load_img(os.path.join(train_folderpath,image_path))



print(f"Las imágenes tienen tamaño: {img_normal.size}")



plt.imshow(img_normal)

plt.title("Normal")

plt.show()



img_name = '0864_bacteria.jpg'

image_path= f'pneumonia/{img_name}'

img_bacteria = load_img(os.path.join(train_folderpath,image_path))

plt.imshow(img_bacteria)

plt.title("Neumonía bacteriana")

plt.show()





img_name = '3799_virus.jpg'

image_path= f'pneumonia/{img_name}'

img_virus = load_img(os.path.join(train_folderpath,image_path))

plt.imshow(img_virus)

plt.title("Neumonía virósica")

plt.show()

from keras.preprocessing.image import ImageDataGenerator





# Tamaño objetivo para escalar las imágenes. 

h,w,c = 32, 32, 3



batch_size=32



# Preprocesamiento de cada subconjunto

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    )



val_datagen = ImageDataGenerator(

    rescale=1. / 255,

    )



test_datagen = ImageDataGenerator(

    rescale=1. / 255,

    )



# Generadores de los subconjuntos. Reciben un directorio, y 

# cada carpeta del directorio se interpreta como una clase distinta.

# En este caso como cada directorio tiene 2 subdirectorios, NORMAL y PNEUMONIA,

# por ende, habrá dos clases.

# Además, al especificar el "class_mode" como binary, la salida se codifica como un solo valor

# (0 o 1), y no como un vector one-hot de dos elementos.



train_generator = train_datagen.flow_from_directory(

    train_folderpath, # directorio de donde cargar las imagenes (train)

    target_size=(h,w),

    batch_size=batch_size,

    class_mode='binary')



val_generator = val_datagen.flow_from_directory(

    val_folderpath, # directorio de donde cargar las imagenes (val)

    target_size=(h,w),

    batch_size=batch_size,

    class_mode='binary')



test_generator = test_datagen.flow_from_directory(

    test_folderpath,# directorio de donde cargar las imagenes (test)

    target_size=(h,w),

    batch_size=batch_size,

    class_mode=None, # IMPORTANTE ya que los ej de test no tienen clase

    shuffle=False # IMPORTANTE ya que nos importa el orden para el archivo de submission

    )





n_train=train_generator.samples

n_val=val_generator.samples

n_test=test_generator.samples

n_clases=train_generator.num_classes

print(f"Los conjuntos de train, val y test tienen {n_train}, {n_val} y {n_test} ejemplos respectivamente.")

print(f"Los conjuntos de datos tienen {n_clases} clases.")
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



model = Sequential()

model.add(Flatten(input_shape=(h,w,c)))

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('sigmoid'))

print(model.summary())
from keras import backend as K

# Definición de las métricas F1, recall y precision utilizando Keras.



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

val_steps=max(1,n_val // batch_size)

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy",f1_m,precision_m, recall_m])

model.fit_generator(train_generator,

                    steps_per_epoch=n_train // batch_size,

                    epochs=2,

                    validation_data=val_generator,

                    validation_steps=val_steps

                    )
#Evaluar el accuracy del modelo en el conjunto entero de entrenamiento

print("*********** Conjunto de entrenamiento ***********")

train_generator.reset()

asd = model.predict_generator(train_generator,steps=n_train // batch_size)

scores = model.evaluate_generator(train_generator,steps=n_train // batch_size)

for metric,score in zip(model.metrics_names,scores):

    print(f"{metric}: {score:.2f}")



print()

# Evaluar el accuracy del modelo en el conjunto entero de validación

print("*********** Conjunto de validación ***********")

val_generator.reset()

asd = model.predict_generator(val_generator,steps=n_val // batch_size)

scores = model.evaluate_generator(val_generator,steps=n_val // batch_size)

for metric,score in zip(model.metrics_names,scores):

    print(f"{metric}: {score:.2f}")

# predecir sobre el conjunto de test y generar el csv resultante

y_prob = model.predict_generator(test_generator,steps=n_test // batch_size)

# Establecer un umbral

treshold=0.5

# Convertir probabilidades a etiquetas con el umbral

y_pred = (y_prob[:,0]>0.5).astype(int)

# quitar el nombre de la carpeta del nombre de archivo

filenames=[ os.path.basename(f) for f in test_generator.filenames]

# igual cant de archivos que de predicciones

assert(len(y_pred)==len(filenames))



# Generar CSV con las predicciones

import csv

with open('solutions.csv', mode='w') as f:

    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)



    writer.writerow(['Id','Expected'])

    print("Id, Expected")

    for f,y in zip(filenames,y_pred):

        print(f"{f}, {str(y)}")

        writer.writerow([f,str(y)])


