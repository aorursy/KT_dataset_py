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
import cv2

import os

import glob
# Mise en place des labels

map_types = {0: 'street', 1: 'mountain', 2: 'forest', 

        3: 'glacier', 4: 'sea', 5: 'buildings'}

pic_size = 64

batch_size = 32

epochs = 200

num_classes = len(map_types)



print (num_classes)
def load_train_set(path): 

    X, y= [], []

    for class_pictures in map_types:

        img_dir = path+ map_types[class_pictures]

        print (img_dir)

        data_path = os.path.join(img_dir,'*g')

        files = glob.glob(data_path)

        for f1 in files:

            img = cv2.imread(f1)

            img = cv2.resize(img, (100,100))

            X.append(np.array(img))

            y.append(class_pictures)

        length= len(X)

        #Connaitre le nombre d'image pour chaque type

        print("Nombre total d'image après insetion du type : ", map_types[class_pictures] , length)

    X = np.array(X)

    y = np.array(y)

    return X, y
X, y = load_train_set("/kaggle/input/intel-image-classification/seg_train/seg_train/")
n_street = 2382

n_moutain = 4894-n_street

n_forest = 7165-4894

n_glacier = 9569-7165

n_sea = 11843-9569

n_buildings = 14034-11843



print("nombre de rue :",n_street)

print("nombre de montagne :",n_moutain)

print("nombre de forêt :",n_forest)

print("nombre de glacier :",n_glacier)

print("nombre de mer :",n_sea)

print("nombre de batiments :",n_buildings)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(X[n_street+n_forest+n_moutain+i])

    plt.title('Label: %s' % map_types[y[n_street+n_forest+n_moutain+i]])
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Normalisation entre 0 et 1

X_train = X_train / 255

X_test = X_test / 255
y_train1 = to_categorical(y_train)

y_test1 = to_categorical(y_test)
y_test1
# Réseau convolutionnel simple

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))



# Compilation du modèle

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=2000, verbose=1)
# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

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

y_cnn = model.predict_classes(X_test)
cm = confusion_matrix(y_cnn,y_test)

print(cm)

plt.figure(figsize = (12,10))
plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

i=1

for j in range(len(X_test)) :

    if (y_cnn[j] != y_test[j]) & (i<50):

        plt.subplot(10,5,i)

        plt.axis('off')

        plt.imshow(X_test[j])

        pred_classe = y_cnn[j].argmax(axis=-1)

        plt.title('%s / %s' % (map_types[int(y_cnn[j])], map_types[int(y_test[j])]))

        i+=1
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

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=50, batch_size=2000, verbose=1)



# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
model.save('mnist_cnn2.h5')
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

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=2000, verbose=1)



# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
for layer in vgg16.layers[15:]:

    layer.trainable=True

for layer in vgg16.layers[0:15]:

    layer.trainable=False
# Compilation du modèle

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Apprentissage

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=2000, verbose=1)



# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
y_cnn = model.predict_classes(X_test)
plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

i=1

for j in range(len(X_test)) :

    if (y_cnn[j] != y_test[j]) & (i<50):

        plt.subplot(10,5,i)

        plt.axis('off')

        plt.imshow(X_test[j])

        pred_classe = y_cnn[j].argmax(axis=-1)

        plt.title('%s / %s' % (map_types[int(y_cnn[j])], map_types[int(y_test[j])]))

        i+=1
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

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=2000, verbose=1)



# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
for i in range (len(ResNet50V2.layers)):

    print (i,ResNet50V2.layers[i])
for layer in vgg16.layers[150:]:

    layer.trainable=True

for layer in vgg16.layers[0:150]:

    layer.trainable=False
# Compilation du modèle

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Apprentissage

train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=2000, verbose=1)



# Test

scores = model.evaluate(X_test, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)