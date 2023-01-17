import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # to plot the images



from sklearn.metrics import accuracy_score



import keras 

from keras.callbacks import EarlyStopping, ModelCheckpoint 

from keras.layers import Input, Dense, Flatten,Dropout

from keras.models import Model

from keras.optimizers import Adam

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator



from keras.applications.vgg16 import VGG16



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#ici on a importé les bibliotheques necessaires
from keras.datasets import cifar10



(X_train,y_train),(X_test,y_test) = cifar10.load_data()
print(X_train.shape)

print(y_train.shape)
print(X_test.shape)

print(y_test.shape)
X_train[0]
y_train
# pour voir le nombre des classes ou des categories

no_of_classes = len(np.unique(y_train))

no_of_classes
y_train_ohe=keras.utils.to_categorical(y_train,no_of_classes)

y_test_ohe=keras.utils.to_categorical(y_test,no_of_classes)
#temps de redimensionner de sorte que toutes les valeurs de pixels se situent entre 0 et 1

X_test = X_test.astype('float32')



X_train=(X_train/255)

X_test=(X_test/255)
X_train[0]
Input_shape = X_train.shape[1:]

Input_shape
# maintenant on va visualiser les 25 premières images de l'ensemble d'entraînement

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(25,5))

for i in range(25):

    ax = fig.add_subplot(5,10,i+1)

    ax.imshow(np.squeeze(X_train[i]))
def create_cnn_model():

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    

    #Create your own input format

    keras_input = Input(shape = Input_shape, name = 'image_input')

    

   #Utilisez le modèle généré #Utilisez le modèle généré

    output_vgg16_conv = model_vgg16_conv(keras_input)

    

    #Ajoutez les couches entièrement connectées 

    x = Flatten(name='flatten')(output_vgg16_conv)

    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu", name='fc1')(x)

    x = Dense(64, activation="relu", name='fc2')(x)

    x = Dense(10, activation='softmax', name='predictions')(x) 

    model = Model( keras_input, x )

    return model



model = create_cnn_model()

model.summary()
datagen = ImageDataGenerator(

    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
model.compile( loss='categorical_crossentropy' , optimizer=Adam(), metrics=['accuracy']) #Configuration du modèle pour la formation à l'aide de la compilation

    

es = EarlyStopping(patience=10, monitor='val_accuracy', mode='max') #Arrêtez l'entraînement lorsqu'une quantité surveillée a cessé de s'améliorer.

#Enregistrez le modèle après chaque époque.

#le dernier meilleur modèle en fonction de la quantité surveillée ne sera pas écrasé.

mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    

#Forme le modèle sur les données générées lot par lot par un générateur Python (ou une instance de Sequence).  

model.fit_generator(datagen.flow(X_train, y_train_ohe), steps_per_epoch=50000/32, validation_data=[X_test, y_test_ohe], callbacks = [es,mc], epochs=1000)

    

model.load_weights('./weights.h5')#nous chargeons les meilleurs poids enregistrés par le ModelCheckpoint

#On charge les meilleurs poids sauvegardés par le ModelCheckpoint



#prédire le test

preds = model.predict(X_test)

test = model.evaluate(X_test,y_test_ohe)

score_test = accuracy_score( y_test, np.argmax(preds, axis=1) )

print (' LE SCORE DE TEST : ', score_test)



preds_classes = np.argmax(preds,axis=-1)

L=7

W=7

fig,axes=plt.subplots(L,W,figsize=(12,12))

axes=axes.ravel()



for i in np.arange(0,L*W):

    axes[i].imshow(X_test[i])

    axes[i].set_title('Prediction= {}\nTrue={}'.format(preds_classes[i],y_test[i]))

    axes[i].axis('off')

    plt.subplots_adjust(wspace=1)