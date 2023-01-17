# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline

# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets
from keras.datasets import mnist

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import os
import glob
monkeys_labels = {0: 'n0', 1: 'n1', 2: 'n2', 3: 'n3', 4: 'n4', 5: 'n5', 6:'n6', 7:'n7', 8:'n8', 9:'n9'}
pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(monkeys_labels)

print (num_classes)
def load_train_set(path): 
    X, Y= [], []
    for class_pictures in monkeys_labels:
        img_dir = path+ monkeys_labels[class_pictures]
        print (img_dir)
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        for f1 in files:
            img = cv2.imread(f1)
            img = cv2.resize(img, (100,100))
            X.append(np.array(img))
            Y.append(class_pictures)
        length= len(X)
        #Connaitre le nombre d'image pour chaque type
        print("Nombre total d'image après insetion du type : ", monkeys_labels[class_pictures] , length)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
X_train, Y_train = load_train_set("/kaggle/input/10-monkey-species/training/training/")
n_n0 = 105
print("Nombre n0 = ", n_n0)
n_n1 = 216 - n_n0
print("Nombre n1 = ", n_n1)
n_n2 = 326 - n_n1 - n_n0
print("Nombre n2 = ", n_n2)
n_n3 = 448 - n_n2 - n_n1 - n_n0
print("Nombre n3 = ", n_n3)
n_n4 = 553 - n_n3 - n_n2 - n_n1 - n_n0
print("Nombre n4 = ", n_n4)
n_n5 = 666 - n_n4 - n_n3 - n_n2 - n_n1 - n_n0
print("Nombre n5 = ", n_n5)
n_n6 = 772 - n_n5 - n_n4 - n_n3 - n_n2 - n_n1 - n_n0
print("Nombre n6 = ", n_n6)
n_n7 = 886 - n_n6 - n_n5 - n_n4 - n_n3 - n_n2 - n_n1 - n_n0
print("Nombre n7 = ", n_n7)
n_n8 = 992 - n_n7 - n_n6 - n_n5 - n_n4 - n_n3 - n_n2 - n_n1 - n_n0
print("Nombre n8 = ", n_n8)
n_n9 = 1098 - n_n8 - n_n7 - n_n6 - n_n5 - n_n4 - n_n3 - n_n2 - n_n1 - n_n0
print("Nombre n9 = ", n_n9)


plt.figure(figsize=(10,20))
for i in range(0,49) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(X_train[i])
    plt.title('Label: %s' % monkeys_labels[Y_train[i]])
X_train.shape
X_test, Y_test = load_train_set("/kaggle/input/10-monkey-species/validation/validation/")
plt.figure(figsize=(10,20))
for i in range(0,20) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(X_test[i])
    plt.title('Label: %s' % monkeys_labels[Y_test[i]])
# Normalisation entre 0 et 1
X_train = X_train / 255
print(X_train[0][0])
# Normalisation entre 0 et 1
X_test = X_test / 255
print(X_test[0][0])
Y_train1 = to_categorical(Y_train)
Y_test1 = to_categorical(Y_test)
# Réseau convolutionnel simple
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(100, 100, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(1)) #1 car on a que 2 sortie possible => malade ou pas malade

# Compilation du modèle
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage
train = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=200, verbose=1)
# Test
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
print(train.history['accuracy'])
print(train.history['val_accuracy'])
def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()
plot_scores(train)
# Prediction
Y_cnn = model.predict_classes(X_test)
cm = confusion_matrix(Y_cnn,Y_test)
print(cm)
plt.figure(figsize = (12,10))
plt.figure(figsize=(15,25))
n_test = X_test.shape[0]
i=1
for j in range(len(X_test)) :
    if (Y_cnn[j] != Y_test[j]) & (i<50):
        plt.subplot(10,5,i)
        plt.axis('off')
        plt.imshow(X_test[j])
        pred_classe = Y_cnn[j].argmax(axis=-1)
        plt.title('%s / %s' % (monkeys_labels[int(Y_cnn[j])], monkeys_labels[int(Y_test[j])]))
        i+=1
# Réseau convolutionnel simple
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(1)) #1 car on a que 2 sortie possible => malade ou pas malade

# Compilation du modèle
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage
train = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=200, verbose=1)
# Test
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
# Modèle CNN plus profond
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #divise les images par 2 (taille)
model.add(Dropout(0.2)) #20% des neurones (aléatoire) ne vont pas apprendre ==> réduit le sur-apprentissage
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten()) #applati
model.add(Dense(512, activation='relu')) #ici neurone ordinaire et pas convolutionnel
model.add(Dense(num_classes, activation='softmax')) #pour classer les images avec le nombre de classe

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage
train = model.fit(X_train, Y_train1, validation_data=(X_test, Y_test1), epochs=50, batch_size=2000, verbose=1)

# Test
scores = model.evaluate(X_test, Y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
# Prediction
Y_cnn = model.predict_classes(X_test)
cm = confusion_matrix(Y_cnn,Y_test)
print(cm)
plt.figure(figsize = (12,10))
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
vgg16.trainable = False
vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Apprentissage
train = model.fit(X_train, Y_train1, validation_data=(X_test, Y_test1), epochs=20, batch_size=2000, verbose=1)

# Test
scores = model.evaluate(X_test, Y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
for layer in vgg16.layers[15:]:
    layer.trainable=True
for layer in vgg16.layers[0:15]:
    layer.trainable=False
# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Apprentissage
train = model.fit(X_train, Y_train1, validation_data=(X_test, Y_test1), epochs=20, batch_size=2000, verbose=1)

# Test
scores = model.evaluate(X_test, Y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
for layer in vgg16.layers[10:]:
    layer.trainable=True
for layer in vgg16.layers[0:10]:
    layer.trainable=False
    
# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Apprentissage
train = model.fit(X_train, Y_train1, validation_data=(X_test, Y_test1), epochs=20, batch_size=2000, verbose=1)

# Test
scores = model.evaluate(X_test, Y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
from keras.applications import InceptionV3, ResNet50V2
ResNet50V2 = ResNet50V2(weights='imagenet', include_top=False, input_shape=(100,100,3))
ResNet50V2.trainable = False
ResNet50V2.summary()
model = Sequential()
model.add(ResNet50V2)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Apprentissage
train = model.fit(X_train, Y_train1, validation_data=(X_test, Y_test1), epochs=20, batch_size=2000, verbose=1)

# Test
scores = model.evaluate(X_test, Y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
for i in range (len(ResNet50V2.layers)):
    print (i,ResNet50V2.layers[i])
for layer in vgg16.layers[126:]:
    layer.trainable=True
for layer in vgg16.layers[0:126]:
    layer.trainable=False
# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Apprentissage
train = model.fit(X_train, Y_train1, validation_data=(X_test, Y_test1), epochs=20, batch_size=2000, verbose=1)

# Test
scores = model.evaluate(X_test, Y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
import cv2
import datetime as dt
import glob
import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras import models, layers, optimizers
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.metrics import confusion_matrix
weights = Path('../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
train_dir = Path('../input/10-monkey-species/training/training/')
test_dir = Path('../input/10-monkey-species/validation/validation/')
height=150
width=150
channels=3
batch_size=32
seed=1337


# Training generator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(height,width), 
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  class_mode='categorical')
# Initialize the base model
base_model = Xception(weights=weights,
                      include_top=False,
                      input_shape=(height, width, channels))
base_model.summary()
def extract_features(sample_count, datagen):
    start = dt.datetime.now()
    features =  np.zeros(shape=(sample_count, 5, 5, 2048))
    labels = np.zeros(shape=(sample_count,10))
    generator = datagen
    i = 0
    for inputs_batch,labels_batch in generator:
        stop = dt.datetime.now()
        time = (stop - start).seconds
        print('\r',
              'Extracting features from batch', str(i+1), '/', len(datagen),
              '-- run time:', time,'seconds',
              end='')
        
        features_batch = base_model.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        
        if i * batch_size >= sample_count:
            break
            
    print("\n")
    
    return features,labels
train_features, train_labels = extract_features(1098, train_generator)
test_features, test_labels = extract_features(272, test_generator)
flat_dim = 5 * 5 * 2048
train_features = np.reshape(train_features, (1098, flat_dim))
test_features = np.reshape(test_features, (272, flat_dim))
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=flat_dim))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()
history = model.fit(train_features, 
                    train_labels, 
                    epochs=30,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=callbacks)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()