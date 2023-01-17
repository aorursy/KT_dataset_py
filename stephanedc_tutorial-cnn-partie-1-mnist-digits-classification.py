# -------------------------------

# Import des librairies nécessaires

# -------------------------------



import matplotlib.pyplot as plt       # Plotting

import numpy as np                    # Tableau Multidimensionnel

import pandas as pd                   # Manipulation des Dataframe

import seaborn as sns                 # Librairie de visualisation



# Librairies Scikit Learn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Dataset des chiffres du MNIST

from keras.datasets import mnist



# Librairies Keras pour la constructino du réseau CNN

from keras.models import Model, Sequential

from keras.models import load_model

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import UpSampling2D, Dropout, Dense, Flatten

from keras.callbacks import TensorBoard
# Fonction pour afficher les données matricielles sous forme d'images

def display_image(X, y, n, label=False):

    plt.figure(figsize=(20,2))

    for i in range(10):

        ax = plt.subplot(1, n, i+1)

        plt.imshow(X.values[i].reshape(28,28))

        if label:

            plt.title("Digit: {}".format(y[i]))

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    plt.show()
# Path vers les jeux de données

train_dir = "../input/train.csv"

test_dir = "../input/test.csv"
# Lecture du jeu d'entrainement via pandas à partir d'un fichier csv

df_train = pd.read_csv(train_dir)



# Récupération des informations sur le Dataframe du jeu de données

df_train.info()
df_train.head()
y_train = df_train['label']

X_train = df_train.drop(columns=['label'])
# Les labels prennent les valeurs des classes de chiffres

y_train.head()
# Visualisation de la répartition des labels

sns.set(style='white', context='notebook', palette='deep')

ax = sns.countplot(y_train)
# On affiche les images connues avec leur labels

# X_train.values[0].reshape(28,28)

display_image(X_train, y_train, n=10, label=True)
# 1. Split entre jeu d'entrainement et jeu de validation avec un ratio de 90/10

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
# 2. Reshape des data pour les formatter en 28x28x1

X_train = X_train.values.reshape(-1, 28,28,1)

X_val = X_val.values.reshape(-1, 28,28,1)
# Affichage de la nouvelle shape des données (maintenant sous forme de matrice 28x28)

X_train.shape
# 3. Normalisation des données pour avoir des valeurs de pixels entre 0 et 255

X_train = X_train / 255.0

X_val = X_val / 255.0
# 4. Remplacement des valeurs des labels par des valeurs catégoriques

Y_train  = pd.get_dummies(y_train).values

Y_val  = pd.get_dummies(y_val).values
print("La valeur {} est encodée vers le vecteur {}".format(y_train[0], Y_train[0]))

print("valeur {} transformée en vecteur: {}".format(y_train[20], Y_train[20]))
# Augmentation des images avec un processing :

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

          featurewise_center=False,            # set input mean to 0 over the dataset

          samplewise_center=False,             # set each sample mean to 0

          featurewise_std_normalization=False, # divide inputs by std of the dataset

          samplewise_std_normalization=False,  # divide each input by its std

          zca_whitening=False,                 # apply ZCA whitening

          rotation_range=20,                   # randomly rotate images in the range (degrees, 0 to 180)

          zoom_range = 0.1,                    # Randomly zoom image 

          width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)

          height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)

          horizontal_flip=False,               # randomly flip images

          vertical_flip=False)                 # randomly flip images
# Initialisation 

model = Sequential()



# ------------------------------------

# Couche de Convolution et MaxPooling

# ------------------------------------



# Conv2D : https://keras.io/layers/convolutional/

#     filters : nombres de filtres de convolutions

#     kernel_size : taille des filtres de la fenêtre de convolution 

#     input_shape : taille de l'image en entrée (à préciser seulement pour la première couche)

#     activation  : choix de la fonction d'activation

# BatchNormalisation : permet de normaliser les coefficients d'activation afin de les maintenirs proche de 0 pour simplifier les calculs numériques

# MaxPooling : Opération de maxPooling sur des données spatiales (2D) : voir illustration ci-dessus

# Dropout : permet de désactiver aléatoirement une proportion de neurones (afin d'éviter le surentrainement sur le jeu d'entrainement)



model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(strides=(2,2)))

model.add(Dropout(0.25))
# ------------------------------------

# Classifier (couche entièrement Connectée)

# Voir illustration ci-dessous

# ------------------------------------

# Flatten : conversion d'une matrice en un vecteur plat

# Dense   : neurones

model.add(Flatten())     # Applatissement de la sortie du réseau de convolution

model.add(Dense(units=1024, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(units=1024, activation='relu'))

model.add(Dropout(0.25))

# Couche de sortie : nombre de neurones = nombre de classe à prédire

model.add(Dense(units=10, activation='softmax'))
# Récapitulatif de l'architecture modèle

model.summary()
# Sélection de l'optimiser pour la decente de gradient

from keras.optimizers import Adam

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.0001), metrics=["accuracy"])
# Démarrage de l'entrainement du réseau

hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),

                           steps_per_epoch=1000,             # nombre image entrainement / batch_size

                           epochs=25,                        # nombre de boucle à réaliser sur le jeu de données complet

                           verbose=1,                        # verbosité

                           validation_data=(X_val, Y_val))   # données de validation (X(données) et y(labels))
# Evaluation de la performance du modèle

final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['acc'], color='b')

plt.plot(hist.history['val_acc'], color='r')

plt.show()
# Prédictions et vecteur de probabilité

Y_hat = model.predict(X_val)

Y_hat[0]
# Génération des vecteurs de verité (Y_true) et de prédiction (Y_pred)

Y_pred = np.argmax(Y_hat, axis=1)

Y_true = np.argmax(Y_val, axis=1)
# Génération d'une matrice de confusion pour observer les erreurs

# Toutes les valeurs sortant de la diagonales sont les erreurs de classification

cm = confusion_matrix(Y_true, Y_pred)

print(cm)
# Lecture jeu de test

X_test = pd.read_csv(test_dir)



# Traitement des données de la même façon que pour l'entrainement

# Reshape

X_test = X_test.values.reshape(-1, 28,28,1)

# Normalisation

X_test = X_test / 255.0
# Prédictions sur le jeu de test

Y_hat = model.predict(X_test, verbose=1)

Y_pred = np.argmax(Y_hat, axis=1)
# Affichage des images prédites

display_image(pd.DataFrame(X_test.reshape(-1, 784)), Y_pred, n=10, label=True)
# Soumission et Enregistrement des résultats

results = pd.Series(Y_pred, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_predictions.csv",index=False)