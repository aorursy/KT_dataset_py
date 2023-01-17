# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for plotting
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Einlesen der Daten train
train = pd.read_csv ("../input/train.csv")

# Einlesen der Daten test
test = pd.read_csv ("../input/test.csv")

# Zuweisung der Variablen
x_train = (train.iloc[:,1:].values).astype('float32')    # alle Pixelwerte ab 1. Spalte 
y_train = train.iloc[:,0].values.astype('int32')         # nur die Labels
x_test = test.values.astype ('float32')

# Normalisieren der Daten
x_train = x_train/255.0
x_test = x_test/255.0

# Ausgabe der Anzahl der Abbildungen
print (x_train.shape[0], 'Train Abbildungen')     # gibt Anzahl der Zeilen von train aus
print (x_test.shape[0], 'Test Abbildungen')       # gibt Anzahl der Zeilen von test aus 
# Umformen der Abbildungen an die Erwartungen von Keras 28x28 Pixel, 1 Kanal
# train und test Abbildungen sind 1D Vektoren mit 784 Werten, für Keras werden diese in 3D Matrizen 28x28x1 umgeformt
X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Importieren von Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

batch_size = 64           # Gesamte Anzahl der training Beispiele in einem einzigen Batch
num_classes = 10          # Anzahl der zu erkennenden Zahlen: 0-9 
epochs = 20               # 1 epoch: ein gesamtes Datenset durchläuft das neuronale Netzwerk einmal
input_shape = (28, 28, 1)
# Konvertieren der Labels in binaere Matrix (One Hot Encoding), z.B. 2 -> [0,0,1,0,0,0,0,0,0,0]
y_train = keras.utils.to_categorical (y_train, num_classes)

# train set wird in zwei Teile geteilt: 10% validation set, 90% für das Training des Modells
X_train, X_val, Y_train, Y_val = train_test_split (X_train, y_train, test_size = 0.1, random_state=42)
# Definieren des Modells
# Sequential Model = linear stack of layers 
# Modell ohne Verzweigungen, jeder Layer hat einen Input und einen Output; der Output von einem Layer ist der Input des Layers darunter
model = Sequential ()

# Convolutional layer: Ein Set lernfähiger Filter -> jeder Filter transformiert einen Teil des Bildes (kernel size) mittels kernel filter
model.add (Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
model.add (Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'))

# Pooling layer: Downsampling Filter -> betrachtet die 2 Nachbarpixel und nimmt den maximalen Wert -> reduziert Rechenkosten und Overfitting
model.add (MaxPool2D((2, 2)))

# Dropout Layer: Regularisierungstechnik -> zufällig ausgewählte Neuronen werden während des Trainings ignoriert -> reduziert Overfitting
model.add (Dropout(0.20))

model.add (Conv2D(64,(3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add (Conv2D(64,(3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add (MaxPool2D(pool_size=(2, 2)))
model.add (Dropout(0.25))
model.add (Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add (Dropout(0.25))

# Flatten layer: konvertiert die finalen feature maps in einen 1D Vektor 
model.add (Flatten())

# Dense Layer: jedes Neuron wird mit dem Neuron im nächsten Layer verknüpft
model.add (Dense(128, activation='relu'))

# Batch Normalization: Jedes Batch wird normalisiert
model.add (BatchNormalization())
model.add (Dropout(0.25))

# gibt die Verteilung der Wahrscheinlichkeiten der einzelnen 10 Klassen aus
model.add (Dense(num_classes, activation='softmax'))

# kompiliert das Modell
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

# Reduzierung der Learning Rate, um globales Minimum der loss function zu finden
learning_rate_reduction = ReduceLROnPlateau (monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.0001)

# Data Augmentation, um Overfitting zu vermeiden
generated_data = ImageDataGenerator (
              featurewise_center=False,
              samplewise_center=False,
              featurewise_std_normalization=False,
              samplewise_std_normalization=False,
              zca_whitening=False,
              rotation_range=15,     # zufällige Rotation der Abbildungen um 15 Grad
              zoom_range = 0.1,      # zufälliger Zoom der Abbildung um 10%
              width_shift_range=0.1,     # zufällige horizontale Verschiebung um 10% der Breite
              height_shift_range=0.1,    # zufällige vertikale Verschiebung um 10% der Höhe
    
              # horizontale und vertikale Spiegelung des Bildes: problematisch bei den Zahlen 6 und 9
              horizontal_flip=False, 
              vertical_flip=False)       
            
model.summary()
# Trainiert das Modell für eine bestimmte Anzahl an epochs (Iterationen eines Datensets)
generated_data.fit(X_train)

# Trainiert das Modell anhand von Daten, die vom Python Generator mit jedem Batch generiert wurden
# ermöglicht real-time data augmentation von Abbildungen in der CPU, während parallel das Modell in der GPU trainiert wird
h = model.fit_generator(generated_data.flow(X_train, Y_train, batch_size=batch_size),
                               epochs = epochs, validation_data = (X_val, Y_val),
                               verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                                , callbacks=[learning_rate_reduction],)
# Evaluation
final_loss, final_accuracy = model.evaluate(X_val, Y_val, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_accuracy))
# Vorhersage der test Abbildungen
prediction = model.predict (X_test)

# gibt den Index bzw. das Label mit dem maximalen Wert zurueck
prediction = np.argmax (prediction, axis = 1)

prediction = pd.Series (prediction, name = "Label")

print (prediction)
# Zeichne 10. test Abbildung
t = plt.imshow (X_test[10] [:,:,0])
# Submission erstellen
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),prediction],axis = 1)

submission.to_csv ("mnist_submission.csv", index=False, header = True)