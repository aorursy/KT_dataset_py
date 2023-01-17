# Este entorno de Python 3 es similar a Jupyter Notebook

# Viene con varias librerías instaladas. Para más információn podés consultar 

# la imagen de docker que utiliza (https://github.com/kaggle/docker-python)



import numpy as np 

import pandas as pd

import keras

import matplotlib.pyplot as plt

%matplotlib inline

import os

from keras.preprocessing.image import load_img

dataset_folderpath="/kaggle/input/aa2019unlp/chest_xray"

print("Hay 3 directorios con los tres subconjuntos de datos: ",os.listdir(dataset_folderpath))

print("La carpeta de cada subconjunto tiene dos subcarpetas: NORMAL y PNEUMONIA")



train_folderpath = os.path.join(dataset_folderpath,"train")

val_folderpath = os.path.join(dataset_folderpath,"val")

test_folderpath = os.path.join(dataset_folderpath,"test")
from keras.preprocessing.image import ImageDataGenerator

# mismo preprocesamiento que el usado originalmente para entrenar MobileNet

from keras.applications.mobilenet import preprocess_input



# Tamaño objetivo para escalar las imágenes. 

h,w,c = 224, 224, 3 # mismo tamaño que el usado originalmente para entrenar MobileNet



batch_size=32



# Preprocesamiento de cada subconjunto

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input # mismo preprocesamiento que el usado originalmente para entrenar MobileNet

    )



val_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input # mismo preprocesamiento que el usado originalmente para entrenar MobileNet

    )



test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input # mismo preprocesamiento que el usado originalmente para entrenar MobileNet

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
from keras.models import Sequential,Model

from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.mobilenet import MobileNet



base_model=MobileNet(input_shape=(224,224,3),weights='imagenet',include_top=False) 

for layer in base_model.layers:

    layer.trainable=False # capas “congeladas” no se entrenan

output = GlobalAveragePooling2D()(base_model.output)    

# Utilizar salida del modelo como entrada a capa Dense de 128 

output=Dense(128,activation='relu')(output)

# Nueva capa de salida

output=Dense(1,activation='sigmoid')(output)



# Crear nuevo modelo en base a lo anterior

model=Model(inputs=base_model.input,outputs=output)



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





def save_predictions(y_prob,filenames,treshold,output="solutions.csv"):

    # Convertir probabilidades a etiquetas con el umbral

    y_pred = (y_prob[:,0]>0.5).astype(int)

    # quitar el nombre de la carpeta del nombre de archivo

    filenames=[ os.path.basename(f) for f in filenames]

    # igual cant de archivos que de predicciones

    assert(len(y_pred)==len(filenames))

    # Generar CSV con las predicciones

    import csv

    with open(output, mode='w') as f:

        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['Id','Expected'])

        print("Id, Expected")

        for f,y in zip(filenames,y_pred):

            print(f"{f}, {str(y)}")

            writer.writerow([f,str(y)])



# predecir sobre el conjunto de test y generar el csv resultante

y_prob = model.predict_generator(test_generator,steps=n_test // batch_size)

# Establecer un umbral

treshold=0.5

save_predictions(y_prob,test_generator.filenames,treshold)
