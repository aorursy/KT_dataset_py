# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720



from IPython.display import FileLink # para tener el link descargable ahí directamente

import numpy as np 

import pandas as pd

import tensorflow as tf

from tensorflow import keras

#import keras

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import metrics

import numpy as np

%matplotlib inline



import os

import warnings

warnings.filterwarnings('ignore')
# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Cualquier resultado que guarden en el directorio actual queda como un "output" en el workspace
# algunas utils

def get_now_as_string():

    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")



def plot_curve(history, ind):

# summarize history for loss

    plt.plot(history.history[ind])

    if ("val_"+ind in history.history.keys()):

          plt.plot(history.history["val_"+ind])

    # plt.plot(history.history['val_loss'])

    plt.title(f'model {ind}')

    plt.ylabel(ind)

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
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
# constantes

# Tamaño objetivo para escalar las imágenes. 

h,w,c = 32, 32, 3

rescale =1./255

batch_size = 32

class_mode = 'binary'

rotation_range = 10



params = {

#     rotation_range: rotation_range, 

    rescale: rescale

}
from keras.preprocessing.image import ImageDataGenerator
# Preprocesamiento de cada subconjunto

train_datagen = ImageDataGenerator(rescale=rescale)

val_datagen = ImageDataGenerator(rescale=rescale)

test_datagen = ImageDataGenerator(rescale=rescale)
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

    class_mode=class_mode

)


val_generator = val_datagen.flow_from_directory(

    val_folderpath, # directorio de donde cargar las imagenes (val)

    target_size=(h,w),

    batch_size=batch_size,

    class_mode=class_mode

)
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
# otras constantes

train_steps = n_train // batch_size

val_steps=max(1, n_val // batch_size)

train_steps, val_steps
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

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input,InputLayer

from keras.layers import Activation, Dropout, Flatten, Dense



## está funcion me retorna modelos. 

## podemos usarla junto a sklearn - KerasClassfier para encontrar los mejores paramétros

def create_model(

    feature_maps=64, 

    kernel_size=3, 

    conv_stride=1, 

    max_pooling=True, 

    pooling_stride=1,#None,  

    activation_max_pooling='relu',

    dropout_rate=0.8,

    **kwargs):

    

    pooling_stride = None if pooling_stride==0 else pooling_stride

    

    model = Sequential()

    model.add(InputLayer(input_shape=(h,w,c)))

    

    #convolutionar layers

    for i in range(2):

        model.add(

            Conv2D(

                int(feature_maps/2),

                kernel_size=(kernel_size, kernel_size), 

                strides=(conv_stride, conv_stride),

                padding='same', # padding zero

                activation=activation_max_pooling

            )

        )



        model.add(MaxPooling2D(strides=pooling_stride))

#     model.add(Dropout(dropout_rate))



    for i in range(3):

        model.add(

            Conv2D(

                filters=feature_maps,

                kernel_size=(kernel_size, kernel_size), 

                strides=(conv_stride, conv_stride),

                padding='same',

                activation=activation_max_pooling

            )

        )



        model.add(MaxPooling2D(strides=pooling_stride))

    model.add(Dropout(dropout_rate))

#     end convolutionar layers



    model.add(Flatten())



    model.add(Dense(512, activation='relu'))

    

    model.add(Dense(1, activation='sigmoid')) # acá va classes



    model.compile(

        loss="binary_crossentropy",

        optimizer="adam",

        metrics=["accuracy", f1_m, precision_m, recall_m]

    )



    return model
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input,InputLayer

from keras.layers import Activation, Dropout, Flatten, Dense



def create_model_1(

    feature_maps=64, 

    kernel_size=3, 

    conv_stride=1, 

    max_pooling=True, 

    pooling_stride=2,#None,  

    activation_max_pooling='relu',

    dropout_rate=0.8,

    **kwargs):

        

    pooling_stride = None if pooling_stride==0 else pooling_stride

    

    model = Sequential()

    model.add(InputLayer(input_shape=(h,w,c)))

    

    model.add(Conv2D(

        filters=feature_maps,

        kernel_size=(kernel_size, kernel_size), 

        strides=(conv_stride, conv_stride),

        padding='same',

        activation=activation_max_pooling

    ))

    model.add(MaxPooling2D(strides=pooling_stride))

    model.add(Dropout(dropout_rate))

    

    model.add(Conv2D(

        filters=feature_maps,

        kernel_size=(kernel_size, kernel_size), 

        strides=(conv_stride, conv_stride),

        padding='same',

        activation=activation_max_pooling

    ))



    model.add(MaxPooling2D(strides=pooling_stride))

    model.add(Dropout(dropout_rate))

    

    model.add(Flatten())

    

    model.add(Dense(512, activation='relu'))

#     model.add(Dropout(dropout_rate))

    

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(

        loss="binary_crossentropy",

        optimizer="adam",

        metrics=["accuracy", f1_m,precision_m, recall_m]

    )

    

    return model
keras.backend.clear_session()



model = create_model()

# model = create_model_1()
es1 = keras.callbacks.EarlyStopping(

    monitor='val_f1_m', 

    patience=30,

    verbose=2, 

    mode='max', 

    restore_best_weights=True

)



es2 = keras.callbacks.EarlyStopping(

    monitor='val_loss', 

    patience=30,

    verbose=2, 

    mode='auto', 

    restore_best_weights=True

)



reduce_lr = keras.callbacks.ReduceLROnPlateau(

    monitor='val_f1_m', 

    factor=0.1,

    patience=2, 

    min_lr=0.000001, 

    verbose=1

)



history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=150,

    validation_data=val_generator,

    validation_steps=val_steps,

    callbacks = [

        es1,

#         reduce_lr

#         es2

    ],

)
# obtenemos y_train, y_val para hacer nuestras pruebas fuera de la herramienta

# este método no era necesario. Generator.classes te las retorna

def obtain_labels(generator, steps):

    y = np.array([])

    i = 0

    for k,v in val_generator:

        y = np.append(y, [*v])

        if (i == (steps-1) or y.shape[0] == generator.samples):

            break

        i += 1

    return y
def get_classes_from_proba(prediction, treshold=0.5):

    # Convertir probabilidades a etiquetas con el umbral

    return (prediction[:,0] > treshold).astype(int)

y_val = val_generator.classes

y_train = train_generator.classes

class_labels = list(train_generator.class_indices.keys())



y_val.shape, y_train.shape, class_labels
#Evaluar el accuracy del modelo en el conjunto entero de entrenamiento

print("*********** Conjunto de entrenamiento ***********")

train_generator.reset()

y_train_proba_pred = model.predict_generator(train_generator,steps=n_train // batch_size)



scores = model.evaluate_generator(train_generator,steps=n_train // batch_size)

for metric,score in zip(model.metrics_names,scores):

    print(f"{metric}: {score:.2f}")

    

y_train_pred = get_classes_from_proba(y_train_proba_pred)
# Evaluar el accuracy del modelo en el conjunto entero de validación

print("*********** Conjunto de validación ***********")

val_generator.reset()

y_val_proba_pred = model.predict_generator(val_generator,steps=n_val // batch_size)



scores = model.evaluate_generator(val_generator,steps=n_val // batch_size)

for metric,score in zip(model.metrics_names,scores):

    print(f"{metric}: {score:.2f}")

    

y_val_pred = get_classes_from_proba(y_val_proba_pred)
print(

    metrics.classification_report(

        y_train.reshape(-1,1), 

        y_train_pred.reshape(-1,1), 

        target_names=class_labels

    )

)
print(

    metrics.classification_report(

        y_val.reshape(-1,1), 

        y_val_pred.reshape(-1,1), 

        target_names=class_labels

    )

)
# plot_curve(history, 'loss')

# plot_curve(history, 'accuracy')

# plot_curve(history, 'f1_m')
# from xgboost import XGBClassifier

# clf = XGBClassifier(n_estimators=10000)

# clf.fit(y_train_proba_pred.reshape(-1, 1), y_train_pred.reshape(-1, 1))

# print(metrics.classification_report(

#     y_train.reshape(-1, 1),

#     clf.predict(y_train_pred.reshape(-1, 1))

# ))

# print(metrics.classification_report(

#     y_val.reshape(-1, 1),

#     clf.predict(y_val_pred.reshape(-1, 1))

# ))

# y_pred = clf.predict(y_prob[:,0].reshape(-1, 1)).astype(int)
# buscamos el mejor treshold (cuando funcione lo de metrics_classification)

for t in [0.45, 0.50, 0.55, 0.60]:

    print(80*"=")

    print(t)

    y_train_p_t = get_classes_from_proba(y_train_proba_pred, t)

    y_val_p_t = get_classes_from_proba(y_val_proba_pred, t)

    print(metrics.classification_report(y_train.reshape(-1, 1), y_train_p_t))

    print(metrics.classification_report(y_val.reshape(-1, 1), y_val_p_t))

    print(80*"=")
# predecir sobre el conjunto de test y generar el csv resultante

y_prob = model.predict_generator(

    test_generator,

    steps=n_test // batch_size

)

# y_prob.shape, y_prob[:,0]
# Establecer un umbral de arriba

treshold=0.5



# Convertir probabilidades a etiquetas con el umbral

y_pred = get_classes_from_proba(y_prob, treshold)
# quitar el nombre de la carpeta del nombre de archivo

filenames=[ os.path.basename(f) for f in test_generator.filenames]

# igual cant de archivos que de predicciones

assert(len(y_pred)==len(filenames))
# solution = pd.DataFrame([], columns=['Id', 'Expected'])

solution=[]

for f,y in zip(filenames,y_pred):

    solution.append([f, str(y)])

solution = pd.DataFrame(solution, columns=['Id', 'Expected'])
file='treshold_5_solutions-{}.csv'.format(get_now_as_string())

solution.to_csv(file, header=True, index=False)
# para descargar el archivo

FileLink(file)


# # Para guardar el modelo y recuperarlo

# import pickle

# pickle_file = 'model_{}.pickle'.format(file)



# with open(pickle_file, 'wb') as handle:

#     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
# # para leer el archivo

# with open(pickle_file, 'rb') as handle:

#     b = pickle.load(handle)

    
# para descargar el archivo

# FileLink(pickle_file)
model.summary()