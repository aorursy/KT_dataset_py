# Import des librairies nécessaires

import matplotlib.pyplot as plt       # Plotting

import numpy as np                    # Tableau Multidimensionnel

import pandas as pd                   # Manipulation des Dataframe

import seaborn as sns                 # Librairie de visualisation



# Librairies Scikit Learn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Dataset des chiffres du MNIST

from keras.datasets import mnist



# Librairies Keras pour la construction du réseau CNN

import keras

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.models import load_model

import matplotlib.pyplot as plt
# Configurations principales de nos modèles

IMG_SIZE          = 28               # taille coté final d'une image en pixel (ici 28x28)

NB_EPOCHS_DENOISE = 10               # nombre epoch alogithme debruiter

NB_EPOCHS_CLASSIF = 10               # nombre epoch alogithme classification des digits

BATCH_SIZE        = 64               # taille batch de traitement

NOISE_FACTOR      = 0.75             # facteur de bruitage gaussian

PLOT_SIZE         = (20,2)           # visualisation matplotlib

DISPLAY_IMG       = 10               # visualisation matplotlib

SAV_MODEL_DENOISE = "denoiser.h5"    # sauvegarde du modele de debruitage

SAV_MODEL_PREDICT = "classifier.h5"  # sauvegarde du modele de classification

NUM_CAT_DIGIT     = 10               
# Fonction pour afficher les données matricielles sous forme d'images

def display_image(X, y, n, label=False):

    plt.figure(figsize=(20,2))

    for i in range(n):

        ax = plt.subplot(1, n, i+1)

        plt.imshow(X[i].reshape(IMG_SIZE, IMG_SIZE))

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
# Chargement des données X et y (pour la classification)

y_train = df_train['label']

X_train = df_train.drop(columns=['label'])  # suppression du label



# Nombre de classe de digits

NUM_CAT_DIGIT = y_train.nunique()

print("Il y a {} classes de Digits dans le Dataset".format(NUM_CAT_DIGIT))
# On affiche les images connues avec leur labels

# Pour convertir un dataframe vers un numpy array on utilise .values

display_image(X_train.values, y_train, n=10, label=True)
# 1. Split entre jeu d'entrainement et jeu de validation avec un ratio de 90/10

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
# 2. Reshape des data pour les formatter en 28x28x1 (3Dimensions nécessaires)

X_train = X_train.values.reshape(-1, 28,28,1)

X_val = X_val.values.reshape(-1, 28,28,1)
# 3. Normalisation des données pour avoir des valeurs de pixels entre 0 et 255

X_train = X_train / 255.0

X_val = X_val / 255.0
# 4. Remplacement des valeurs des labels par des valeurs catégoriques

Y_train  = pd.get_dummies(y_train).values

Y_val  = pd.get_dummies(y_val).values
# Lecture jeu de test

X_test = pd.read_csv(test_dir)



# Traitement des données de la même façon que pour l'entrainement

# Reshape

X_test = X_test.values.reshape(-1, 28,28,1)

# Normalisation

X_test = X_test / 255.0
# Processing des images (pas de traitement particulier pour ne pas alourdir le modèle) :

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( featurewise_center=False )
# Initialisation 

classifier = Sequential()

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



classifier.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same', input_shape = (IMG_SIZE, IMG_SIZE, 1)))

classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='Same'))

classifier.add(MaxPooling2D(strides=(2,2)))

classifier.add(Dropout(0.25))
# -----------------------------------------

# Classifier (couche entièrement Connectée)

# -----------------------------------------



# Flatten : conversion d'une matrice en un vecteur plat

# Dense   : neurones

classifier.add(Flatten())     # Applatissement de la sortie du réseau de convolution

classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dropout(0.25))



# Couche de sortie : nombre de neurones = nombre de classe à prédire

classifier.add(Dense(units=NUM_CAT_DIGIT, activation='softmax'))
# Récapitulatif de l'architecture modèle de classification

classifier.summary()
# Sélection de l'optimiser pour la decente de gradient

classifier.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.0001), metrics=["accuracy"])
# Démarrage de l'entrainement du réseau

hist = classifier.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

                                steps_per_epoch=200,              # nombre d'étape car nous utilisons des générateurs

                                epochs=NB_EPOCHS_CLASSIF,         # nombre de boucle à réaliser sur le jeu de données complet

                                verbose=1,                        # verbosité

                                validation_data=(X_val, Y_val))   # données de validation (X(données) et y(labels))
# Evaluation de la performance du classifier

final_loss, final_acc = classifier.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# Affichage des courbes d'apprentissage

# Loss

plt.subplots(figsize=(12,6))

plt.title("Loss")

plt.plot(hist.history['loss'], color='b', label='train_loss')

plt.plot(hist.history['val_loss'], color='r', label='validation_loss')

plt.legend(loc='upper right')

plt.show()



# Accuracy

plt.subplots(figsize=(12,6))

plt.title("Accuracy")

plt.plot(hist.history['acc'], color='b', label='train_accuracy')

plt.plot(hist.history['val_acc'], color='r', label='validation_accuracy')

plt.legend(loc='upper left')

plt.show()
# Prédictions et vecteur de probabilité

Y_hat = classifier.predict(X_test)
# Génération des vecteurs de verité (Y_true) et de prédiction (Y_pred)

Y_pred = np.argmax(Y_hat, axis=1)

Y_true = np.argmax(Y_val, axis=1)
# Affichage des prédictions

display_image(X_test.reshape(-1, 784), Y_pred, n=10, label=True)
# ajout d'un bruit Gaussien

X_train_noisy = X_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

X_val_noisy   = X_val + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_val.shape) 

X_test_noisy  = X_test + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 



# limitation des valeurs entre 0 et 1 (l'ajout de bruit peut provoquer des dépassements)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)

X_val_noisy   = np.clip(X_val_noisy, 0., 1.)

X_test_noisy  = np.clip(X_test_noisy, 0., 1.)
# Visualisation Images bruitées

display_image(X_val_noisy, None, n=10, label=False)
# Création de l'autoencoder



# Création du format d'entrée (Input Tensor)

input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1)) 



# 1 Encode

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

encoded.shape



"""

A ce point les dimensions de l'image sont (7, 7, 32) => 1568 dimensions

Pour ce travail et pour améliorer la reconstruction des images bruyantes nous gardons une 

haute dimensionalité

"""



# 2 Decode

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)



# Un autoencoder utilise la fonction sigmoid comme fonction d'activation

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoded.shape



# Assemblage du modele

autoencoder = Model(input_img, decoded)



# Pour les autoencoders la fonction de cout peut être soit binary_crossentropy soit rmse

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()
# Visualisation des images encodées

encoder = Model(input_img, encoded)

encoded_imgs = encoder.predict(X_train)

n = DISPLAY_IMG

plt.figure(figsize=(20, 8))

for i in range(n):

    ax = plt.subplot(1, n, i+1)

    plt.imshow(encoded_imgs[i].reshape(16, 2 * 49).T)

    plt.title("Encoded: {}".format(y_train[i]))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
# Entrainement de l'autoencoder

autoencoder.fit(X_train_noisy, X_train,

                epochs=NB_EPOCHS_DENOISE,

                batch_size=BATCH_SIZE,

                shuffle=True,

                validation_data=(X_val_noisy, X_val),            

                )
""" decodage et prediction"""

# --- decodage des images bruitées via le réseau de neurones de classification

# --- prédiction de catégorie des images débruitées

decoded_imgs = autoencoder.predict(X_test_noisy)

predicted_decoded_digit = classifier.predict_classes(decoded_imgs)



# -- affichage image originale

# -- et de la reconstruction (debruitage)

n = DISPLAY_IMG * 2

plt.figure(figsize=(PLOT_SIZE))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i+1)

    plt.title("Pred: {}".format(predicted_decoded_digit[i]))

    plt.imshow(X_test_noisy[i].reshape(IMG_SIZE, IMG_SIZE))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # display reconstruction

    ax = plt.subplot(2, n, i+1 + n)

    plt.imshow(decoded_imgs[i].reshape(IMG_SIZE, IMG_SIZE))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()