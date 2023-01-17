import numpy as np

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,Callback

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from keras.constraints import maxnorm

from keras.optimizers import SGD, Adam

from keras.utils import np_utils 

from keras import backend as K 

from keras.models import load_model 

import matplotlib.pyplot as plt 
# importation de données

# x_train:données d'entraînement (images), 

#y_train: étiquettes (chiffres)

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Declarations des variables

nbre_classes = 10 # le nombre de classes

epochs = 100 # nombre d'itérations sur les données

batch_size = 256 # 32 examples in a mini-batch, smaller batch size means more updates in one epoch

# conversionen float et normalisation des entrées de 0-255 à 0-1

from keras.utils import np_utils 

#Application du One Hot Encoding

y_train = np_utils.to_categorical(y_train, nbre_classes)

y_test = np_utils.to_categorical(y_test, nbre_classes)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train  /= 255

x_test /= 255

print(x_train.shape)

print(y_train.shape)
def cifar10Model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3),  input_shape=(32,32,3), activation='relu', padding='same', )) 

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Dropout(0.2)) 

    

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Dropout(0.2)) 

    

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2))) 

    model.add(Dropout(0.2)) 

        

    model.add(Flatten()) 

    model.add(Dense(512, activation='relu')) 

    model.add(Dropout(0.5)) 

    model.add(Dense(nbre_classes, activation='softmax'))

    

    # Compilation du modèle

    sgd = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=False) 

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model



    model=cifar10Model()

    model.summary()
weights_file = 'CIFAR-10.h5'



lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),

                               cooldown=0, patience=10, min_lr=0.5e-6) # ReduceLROnPlateau pour diminuer le taux d'apprentissage

early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5) #Arrête l'entraînement lorsqu'une quantité surveillée a cessé de s'améliorer.

model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc',

                                   save_best_only=True,

                                   save_weights_only=True, mode='auto') # pour enregistrer le modèle après chaque époque.



callbacks = [lr_reducer, early_stopper, model_checkpoint] # Classe de base abstraite utilisée pour créer de nouveaux rappels.
 

# entrainnement du modèle

#CC=CustumCallback()

#callbacks = [CC

history = model.fit(x_train, y_train, batch_size=batch_size,verbose=True, epochs=epochs, 

                    validation_data=(x_test,y_test), callbacks = callbacks)
# Resulat après entrennement

scores = model.evaluate(x_test, y_test, verbose=0) 

print("Accuracy: %.2f%%  loss: %.3f" % (scores[1]*100,scores[0]))
from keras.models import load_model 

model.save('model_cifar10-latest.h5')
#enregistrement sur le disque

model_json = model.to_json()

with open('model-latest.json', 'w') as json_file:

    json_file.write(model_json)

model.save_weights('model-cifar10w-latest.h5') 
#Fonction de lecture des images

def show_imgs(X):

    plt.figure(1)

    k = 0

    for i in range(0,4):

        for j in range(0,4):

            plt.subplot2grid((4,4),(i,j))

            plt.imshow((X[k]))

            k = k+1

    # show the plot

    plt.show()
#Importation du modèle: paramètres enregistrés et test sur quelques images

show_imgs(x_test[:16])

json_file = open('model-latest.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model.load_weights('model-cifar10w-latest.h5')

labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

 

indices = np.argmax(model.predict(x_test[:16]),1)

print([labels[x] for x in indices]) 
