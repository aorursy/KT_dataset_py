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
# Affichage des fichiers dans le répertoire de données Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
img_dir = "/kaggle/input/cat-and-dog/training_set/training_set/cats" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

X=[]

y=[]

for f1 in files:

    img = cv2.imread(f1)

    img = cv2.resize(img, (100,100))

    X.append(np.array(img))

    y.append(0)

n_cats = len(X)
plt.imshow(X[0])

plt.title(y[0])
np.array(X).shape
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(X[i])

    plt.title('Label: %i' % y[i])
img_dir = "/kaggle/input/cat-and-dog/training_set/training_set/dogs" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

for f1 in files:

    img = cv2.imread(f1)

    img = cv2.resize(img, (100,100))

    X.append(np.array(img))

    y.append(1)

n_dogs = len(X)-n_cats
print(n_dogs)

print(n_cats)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(X[n_cats+i])

    plt.title('Label: %i' % y[n_cats+i])
X = np.array(X)

y = np.array(y)
X.shape
# Normalisation entre 0 et 1

X = X / 255

print(X[0][0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Réseau convolutionnel simple

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(100, 100, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(Dense(128, activation='relu'))

model.add(Dense(1))



# Compilation du modèle

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()
# Apprentissage

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

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

        plt.title('%d / %d' % (y_cnn[j], y_test[j]))

        i+=1
# Modèle CNN plus profond

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200, verbose=1)



# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
model.save('mnist_cnn2.h5')
new_model = load_model('mnist_cnn2.h5')

new_model.summary()
scores = new_model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

vgg16.trainable = False
vgg16.summary()
model = Sequential()

model.add(vgg16)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
for i in range (len(vgg16.layers)):

    print (i,vgg16.layers[i])
for layer in vgg16.layers[15:]:

    layer.trainable=True

for layer in vgg16.layers[0:15]:

    layer.trainable=False
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
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

        plt.title('%d / %d' % (y_cnn[j], y_test[j]))

        i+=1
from keras.applications import InceptionV3, ResNet50V2


  
