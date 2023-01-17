#Importo las librerias 

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

np.random.seed(123)

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.utils.np_utils import to_categorical # utilizada para convertir etiquetas a codificación en

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

import itertools

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical 



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
# aqui grafico el modelo de perddias y el modelo de presision

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    #acc=accurancy val_acc=valor de precision

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Modelo de precision')

    axs[0].set_ylabel('Precision')

    axs[0].set_xlabel('Epoch') #epocas

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

   

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Modelo de perdidas')

    axs[1].set_ylabel('Perdidas')

    axs[1].set_xlabel('Epoch')#epocas

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

#aquise hace un diccionario de ruta de imagenes para poder unir las carpetas HAM10000_images_part1 y part2 en la carpeta base_skin_dir



base_skin_dir = os.path.join('..', 'input')



# Fusionar imágenes de ambas carpetas HAM10000_images_part1 y part2 en un diccionario



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}



#aqui cambio las etiquetas que se muestran por el nombre completo



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

##aqui se hace la lectura de los datos y se procesan los datos

#Leemos el dataset HAM10000_metadata.csvuniendolo a la ruta de la carpeta de imagenes base_skin_di que hice arriba



skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))



#Aqui creo nuevs columnas para poder tener los datos image_id, cell_type tiene el nombre corto de la etiquta del tipo de lesión y

#cell_type_idx que asigna una categoria a cada tipo de lesión  va del 0 al 6



skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)

skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 

skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

#para ver como esta el datset modificado

skin_df.head()
#veo los valores nulos qeu tienen las columnas del dataset

skin_df.isnull().sum()
#como en la columna edad hay datos vacios y no son muchos entonves voya  llenarlos con la media

skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
#veo si se hizo el cambio

skin_df.isnull().sum()
#veo que tipo d e dato maneja cada columna

print(skin_df.dtypes)
#aqui grafico los tipos de cancer para ver como estan distribuidos los datos 

fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
#nos indica la forma en la fue diagnosticado el cáncer

skin_df['dx_type'].value_counts().plot(kind='bar')
#Grafica de donde esta unicado

skin_df['localization'].value_counts().plot(kind='bar')
#veo la distribucion de la edad

skin_df['age'].hist(bins=40)
#para ver si son hombres o ujeres los paicentes

skin_df['sex'].value_counts().plot(kind='bar')
# aqui cargo y cambio el tamañod elas imagenes cambio el tamaño por que las dimensiones son de 450 x 600 x3 y tensorflow no tranaja con ese tamaño el tamaño debe ser 100*75

#las imagenes se cargan en la caolumna path(sera la ruta de la imagen) en image se guarda el codigod e la imagen en color rgb



skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
#veo dataset

skin_df.head()
#aqui veo las imagenes 5 por cada tipo de lesion 



n_samples = 5

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))

for n_axs, (type_name, type_rows) in zip(m_axs, 

                                         skin_df.sort_values(['cell_type']).groupby('cell_type')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):

        c_ax.imshow(c_row['image'])

        c_ax.axis('off')

fig.savefig('category_samples.png', dpi=300)
# Comprobando la distribución del tamaño de la imagen

skin_df['image'].map(lambda x: x.shape).value_counts()
features=skin_df.drop(columns=['cell_type_idx'],axis=1)

target=skin_df['cell_type_idx']
#vamos a dividos los datos ccon 80:20

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)
# normalizamos los datos de x_train, x_test restando de sus valores medios y luego dividiendo por su desviación estándar.

x_train = np.asarray(x_train_o['image'].tolist())

x_test = np.asarray(x_test_o['image'].tolist())



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std
#La capa de salida también tendrá dos nodos, por lo que necesitamos alimentar nuestra serie de etiquetas en un 

#marcador de posición para un escalar y luego convertir esos valores en un vector caliente.



y_train = to_categorical(y_train_o, num_classes = 7)

y_test = to_categorical(y_test_o, num_classes = 7)
#aqui se hace la divicion de datos para el entrenamiento y para la valizacion

#se usara 90:10 90 para entrenar y 10 para validadicones



x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)
#reshape reforma la imagen debe tener 3 dimenciones (alto=75 px,ancho=100 px,canal=3)

x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))

x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))

x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))
# modelo

input_shape = (75, 100, 3)

num_classes = 7



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))

model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.40))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
# defino el optimizador

#Adam Optimizer trata de solventar el problema con la fijación de el ratio de aprendizaje 

#para ello adapta el ratio de aprendizaje en función de cómo estén distribuidos los parámetros.

#Si los parámetros están muy dispersos  el ratio de aprendizaje aumentará.



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compilo el modelo

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#para que el optimizador converja más rápido y más cercano al mínimo global de la función de pérdida, se usa el metodo de recocido de la tasa de aprendizaje (LR)

#El LR es el paso por el cual el optimizador recorre el 'panorama de pérdidas'. Cuanto más alto es el LR, más grandes son los pasos y más rápida es la convergencia.

#Sin embargo, el muestreo es muy pobre con un alto LR y el optimizador probablemente podría caer en un mínimo local.

#Con la función ReduceLROnPlateau de Keras.callbacks, se reduce el LR a la mitad si la precisión no mejora después de 3 épocas.



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, #condicon

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
#se hace un aumento de imagen para evitar el sobreajuste, expandimos el dataset ham

#Para evitar problemas de sobreajuste, necesitamos expandir artificialmente nuestro conjunto de datos HAM 10000. Podemos hacer que su conjunto de datos existente sea aún más grande.

#La idea es alterar los datos de entrenamiento con pequeñas transformaciones para reproducir las variaciones. esto puede duplicar o triplicar el número de ejemplos de entrenamiento y crear un modelo muy robusto.

#solo s ehace al 10%

datagen = ImageDataGenerator(

        featurewise_center=False,  # establece lamedia d eerada a 0 sobre el datase

        samplewise_center=False,  # establece cada media de muestra en 0

        featurewise_std_normalization=False,  # divide las entradas por std del dataset

        samplewise_std_normalization=False,  # divide cada entrada por su estandar

        zca_whitening=False,  # aplica blanqueador ZCA este por que hace que la imagen s esiga pareciendo a la original

        rotation_range=10,  # rota imagen al azar en el rango (grados, 0 a 180)

        zoom_range = 0.1, # amplia imagen aleatoriamnete

        width_shift_range=0.1,  # cambio aleatoriamnet ls imagenes a horizontal

        height_shift_range=0.1,  # cambo aleatorimente las imagens a vertical

        horizontal_flip=False,  # voltea imagens al azar horizontamnte

        vertical_flip=False)  # voltea imagens al azar verticalemne



datagen.fit(x_train)

# Ajusto el modelo

epochs = 250 #epocas

batch_size = 10 #lotes

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
#Evaluacion modelo

#aqui se verifica la presicion,se valida el modelo



loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))



model.save('model.h5')

model.save_weights('pesos.h5')



plot_model_history(history)
# Function to plot confusion matrix    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

  

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predecir los valores del conjunto de datos de validación

Y_pred = model.predict (x_validate)

# Convierte las clases de predicciones en vectores calientes

Y_pred_classes = np.argmax (Y_pred, axis = 1)

# Convertir observaciones de validación a vectores calientes

Y_true = np.argmax (y_validate, axis = 1)

# calcular la matriz de confusión

confusion_mtx = confusion_matrix (Y_true, Y_pred_classes)



 



# grafico dema matriz de confusion

plot_confusion_matrix(confusion_mtx, classes = range(7)) 
#en esta grafica vemos que categoria tiene mas predicciones incrrectas

label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)

plt.bar(np.arange(7),label_frac_error)

plt.xlabel('True Label')

plt.ylabel('Fraction classified incorrectly')