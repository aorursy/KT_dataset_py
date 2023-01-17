#IMPORTIEREN VON PANDAS ZUM EINLESEN DER DATEN. ZUSÄTZLICH SCIKIT FÜR DAS SPLITTEN DER DATEN
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
%matplotlib inline

#EINLESEN DER DATEN AUS TRAIN.CSV UND ZUWEISUNG ZU DEN VARIABLEN
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# DAMIT DIE DATEN VON KERAS GENUTZT WERDEN KÖNNEN, MÜSSEN WIR DIE DATEN ZUVOR UMFORMEN [beispiele][breite][höhe][pixel]
train_images = train_images.values.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.values.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

#UMWANDELN DER DATEN VON VECTOR ZU BINARY-matrix
from keras.utils import np_utils
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
num_classes = test_labels.shape[1]
input_shape = (28, 28, 1)

# NORMALISIEREN DER DATEN
train_images /= 255
test_images /= 255
print('Anzahl der Trainings-Bilder', train_images.shape[0])
print('Anzahl der Test-Bilder', test_images.shape[0])


#KERAS IMPORTIEREN
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


#DEFINIEREN DES NEURALES NETZWERKES
def neurales_netzwerk():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(5,5), input_shape=(28, 28, 1),activation='relu')) #INPUTLAYER DES NETZWERKES 
    model.add(MaxPooling2D(pool_size=(2,2))) #POOLINGLAYER - ENTFERNEN ÜBERFLÜSSIGER INFORMATIONEN
    model.add(Dropout(0.15)) #Zufälliges Löschen von neuronen. Overfitting verhindern.
    model.add(Flatten()) #2d zu vector
    model.add(Dense(128,activation='relu')) #NN-Layer mit 128 neuronen
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# DAS MODEL ANHAND DER METHODE ERSTELLEN
model = neurales_netzwerk()

#TRAINIEREN DES MODELS
model.fit(x=train_images,y=train_labels,validation_data=(test_images, test_labels), epochs=10)

#AUSGABE DER GENAUIGKEIT
scores = model.evaluate(test_images, test_labels, verbose=0)
print("Fehlerrate: %.2f%%" % (100-scores[1]*100))

#ANWENDEN DES GELERNTEN MODELLS AUF DIE TESTDATEN UND SPEICHERN ALS CSV

import numpy as np
test_data = pd.read_csv('../input/test.csv')
print('Dataframeangaben', test_data.shape)
test_data = test_data.values.reshape(test_data.shape[0], 28, 28, 1).astype('float32')
test_data = test_data/255

prediction = pd.DataFrame()
imageid = []
for i in range(len(test_data)):
    i = i + 1
    imageid.append(i)
prediction["ImageId"] = imageid 
prediction["Label"] = model.predict_classes(test_data, verbose=0)
print(prediction[:10])
prediction.to_csv("predict.csv", index=False)


#Testen, ob die Vorhersagen einigermaßen stimmen.

image_index = 6
plt.imshow(test_data[image_index].reshape(28, 28),cmap='binary')
pred = model.predict(test_data[image_index].reshape(1, 28, 28, 1))
print('Vorhersage: ', pred.argmax())

