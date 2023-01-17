# Imports necesarios base

import numpy as np 

import os

import cv2

import matplotlib.pyplot as plt

import shap
# Imports necesarios modelo keras

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Activation, BatchNormalization

from keras.optimizers import Adam, SGD

from keras.models import Sequential, Model

from keras.preprocessing import image

from keras import backend as K

from keras.callbacks import EarlyStopping
#Imports especificos del modelo

from keras.applications import VGG16, MobileNet, Xception
#Imports necesarios para k-fold

# Esto es para asegurar siempre los mismos folds

import random

from sklearn.model_selection import StratifiedKFold
#Imports para preprocesar Y

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder

from keras.utils.np_utils import to_categorical
# Directorios del conjunto de entrenamiento y test

train_dir = "../input/train/train"

test_dir = '../input/test_mixed/Test_Mixed'
# Obtenemos las clases disponibles (nombres de las carpetas)

clases = os.listdir(train_dir)

#print("Existen un total de {} clases de corales".format(len(clases)))

clases.sort()

print("Las clases son: {}".format(clases))
# Cargamos el conjunto de datos de entrenamiento en memoria (hay ~17GB en la máquina, nos sobra)

X = np.array([cv2.imread(os.path.join(train_dir, cl, name)) for cl in clases

           for name in os.listdir(os.path.join(train_dir, cl))])

Y = np.array([cl for cl in clases

           for name in os.listdir(os.path.join(train_dir, cl))])

#Leemos el csv del sample submission, porque ahi si vienen ordenados los tests.

import csv

with open('../input/SampleC.csv', 'r') as csvFile:

    reader = csv.reader(csvFile)

    x_test_names = np.array([row[0] for row in reader])



csvFile.close()

x_test_names = np.delete(x_test_names, 0)



Y_original = Y.copy()



X_test = np.array([cv2.imread(os.path.join(test_dir, name)) 

            for name in x_test_names])

for i in range(0,len(clases)):

    for j in range(0,len(Y)):

        if Y[j] == clases[i]:

            Y[j] = i

Y = Y.astype(int)

#print(x_test_names)
#Reshape cutre. estoy quedandome con la esquina superior izq de cada una de las imagenes y recortando





#X = X[:,:224,:224,:]

#X_test  = X_test[:,:224,:224,:]

backgroundX = X.copy()



X = X.astype('float32')

X /= 255

X_test = X_test.astype('float32')

X_test /= 255





#Procesamos Y, binarizamos las clases

#mlb = MultiLabelBinarizer()

Y = to_categorical(Y, 14)
def get_model():

    #Cargamos el modelo de keras preentrenado

    vgg_model = VGG16(weights = 'imagenet',include_top = False,input_shape = X[0].shape)

    #Añadimos las modificaciones al modelo

    #Recuperamos la salida del modelo VGG16

    layer_dict = dict([(layer.name,layer) for layer in vgg_model.layers])



    #print(layer_dict)

    x = layer_dict['block5_pool'].output



    x = Flatten(name = 'batch')(x)

    x = Dense(4096, activation='relu', name='fc1')(x)

    x = Dropout(0.25,name = 'Dropout1')(x)

    x = Dense(4096, activation='relu', name='fc2')(x)

    x = Dropout(0.25,name = 'Dropout2')(x)

    #x = Dense(1024, activation='relu', name='fc3')(x)

    #x = Dropout(0.1,name = 'Dropout3')(x)

    #x = Dense(14, activation='softmax', name='predictions')(x)

    

    #x = Flatten()(x)

    #x = Dense(1024)(x)

    #x = Activation('relu')(x)

    #x = BatchNormalization()(x)

    #x = Dropout(0.1, seed=7)(x)



    #x = Dense(512)(x)

    #x = Activation('relu')(x)

    #x = BatchNormalization()(x)

    # x = Dropout(0.1, seed=seed)(x)



    # Añadimos una última capa completamente conectada con una salida

    x = Dense(14, activation="softmax", name='predictions')(x)



    #x = Dense(128,activation='relu')(x)

    #x = Dense(64,activation='relu')(x)

    #x = Dense(32,activation='relu')(x)

    #x = Dense(14,activation='softmax')(x)

    

    custom_model = Model(input = vgg_model.input, output = x)

    #Bloqueamos el entrenamiento en las 

    for layer in custom_model.layers[:19]:

        if type(layer) is not MaxPooling2D:

            layer.trainable = False

        

    



    custom_model.compile(loss = 'categorical_crossentropy',

            optimizer = Adam(lr = 0.00001,beta_1=0.9, beta_2=0.999, decay=1e-4),

                 metrics = ['accuracy'])

    return custom_model
#Do an histogram equalization to all the images

from skimage import data, img_as_float

from skimage import exposure

X_hist = X.copy()

X_test_hist = X_test.copy()

for x in X_hist:

    for channel in range(x.shape[2]):

        x[:,:,channel] = exposure.equalize_hist(x[:,:,channel]) 



for x_test in X_test_hist:

    for channel in range(x_test.shape[2]):

        x_test[:,:,channel] = exposure.equalize_hist(x_test[:,:,channel])
#Creamos las particiones de los folds

from sklearn.model_selection import train_test_split

random.seed(7)

num_folds = 5

skf = StratifiedKFold(n_splits=num_folds)

avg_score_test = 0

avg_score_train = 0

act_fold = 1

epochs = 200

best_score_test = 0.0

es = EarlyStopping(monitor='val_loss',

                               mode='min',

                               min_delta=1e-4,

                               patience=5,

                               restore_best_weights=True,

                               verbose=1)

best_accuracy = 0

for train_index, dev_index in skf.split(X_hist,Y_original):

    print("Fold: ", act_fold)

    act_fold = act_fold+1

    X_train, X_dev = X_hist[train_index],X_hist[dev_index]

    Y_train, Y_dev = Y[train_index] , Y[dev_index]

    

    custom_model = get_model()

    

    datagen = image.ImageDataGenerator(

        zoom_range = 0.4,

        rotation_range = 10,

        width_shift_range= 0.2,

        height_shift_range=0.2)

    datagen2 = image.ImageDataGenerator(

        zoom_range = 0.4,

        rotation_range = 10,

        width_shift_range= 0.2,

        height_shift_range=0.2)

    

    X_trainTrue, X_test, y_trainTrue, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=7,

                                                               shuffle =True, stratify=Y_train)



    # compute quantities required for featurewise normalization

    # (std, mean, and principal components if ZCA whitening is applied)

    datagen.fit(X_trainTrue)

    datagen2.fit(X_test)



    # fits the model on batches with real-time data augmentation:

    custom_model.fit_generator(datagen.flow(X_trainTrue, y_trainTrue, batch_size=32),

                    steps_per_epoch=len(X_train) / 32, epochs=epochs, validation_data = datagen.flow(X_test, y_test, batch_size=48),

                              callbacks = [es], shuffle=True, validation_steps = 48)

    

    

    print("Rendimiento train:")

    score = custom_model.evaluate(X_train,Y_train)

    print(score)

    print("Rendimiento dev:")

    score_test = custom_model.evaluate(X_dev,Y_dev)

    

    

    avg_score_test = avg_score_test + score_test[1]

    avg_score_train = avg_score_train+score[1]

    print(score_test)

    #Vamos a mostrar en lo que se fija el modelo

    #background = backgroundX[np.random.choice(backgroundX.shape[0], 100, replace=False)]

    #e = shap.DeepExplainer(custom_model, background)

    #shap_values = e.shap_values(X_dev[1:5])

    #shap.image_plot(shap_values, -X_dev[1:5])

    

    K.clear_session()

    

print("Media en train:",avg_score_train/num_folds,"   Media en test: ", avg_score_test/num_folds)
custom_model = get_model()

datagen = image.ImageDataGenerator(

        zoom_range = 0.4,

        rotation_range = 10,

        width_shift_range= 0.2,

        height_shift_range=0.2)

datagen.fit(X)





#fits the model on batches with real-time data augmentation:

custom_model.fit_generator(datagen.flow(X_hist, Y, batch_size=32),

        steps_per_epoch=len(X) / 32, epochs=27, shuffle=True)

    

score = custom_model.evaluate(X_hist,Y)

print(score)

y_probs = custom_model.predict(X_test_hist)

y_hat = y_probs.argmax(axis=-1)

print(y_hat)
#x_test_names.sort()



f = open("sampleSubmission.csv", "w")

f.write("Id,Category\n")

for i in range(0, len(x_test_names)):

    f.write(str(x_test_names[i]) + "," + str(y_hat[i]) + '\n')

print(y_hat)