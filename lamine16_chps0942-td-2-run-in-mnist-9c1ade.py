# Listage des données source
import os
racine_data = "../input";
repertoires = os.listdir(racine_data)
for repertoire in repertoires:
    print(repertoire)
    fichiers = os.listdir(racine_data+"/"+repertoire)
    for fichier in fichiers:
        print("   > " +fichier)
# Chargement des données d'entrainement et de test
import pandas as pd
train = pd.read_csv('../input/digits/train.csv')
test100 = pd.read_csv('../input/test-100-premiers/test100.csv')
evaluation = pd.read_csv('../input/digits/test.csv')
# Affichage des informations
print("Il y a {0} exemples d'apprentissage.".format(train.shape[0]))
print("Il y a {0} exemples d'évaluation.".format(evaluation.shape[0]))
print("Il y a {0} exemples de test.".format(test100.shape[0]))
print("Nombre de colonnes : " , train.shape[1])
print("Liste des colonnes :")
train.columns
# On sépare la classe (label) des données (pixels)

# Le vecteur des numéros de classe
train_label = train["label"]

# Suppression de cette colonne 
train.drop("label", axis = 1 , inplace=True)

# Nombre d'exemples d'entrainement par classe
train_label.value_counts().sort_index()
# La même chose pour les tests
test100_label = test100["label"]
test100.drop("label", axis = 1 , inplace=True)
test100_label.value_counts().sort_index()
# Normalisation des valeurs des images [0-255] -> [0.0-1.0]
train      = train/255.0
test100    = test100/255.0
evaluation = evaluation/255.0

# Affichage des images de l'ensemble de test

import matplotlib.pyplot as plt
%matplotlib inline

# Dimension de l'affichage
nb_cols = 10
nb_ligs = 10
taille  = 20

# Affichage de nb_ligs lignes de nb_cols images
fig1, ax1 = plt.subplots(nb_ligs,nb_cols, figsize=(taille,taille))
for no_lig in range(nb_ligs):
    for no_col in range(nb_cols):
        ax1[no_lig][no_col].imshow(test100.iloc[no_lig*nb_cols+no_col].values.reshape((28,28)), cmap='gray')
        ax1[no_lig][no_col].axis('off')
        ax1[no_lig][no_col].set_title(test100_label[no_lig*nb_cols+no_col])  
import numpy as np
# Restructuration des images en 3 dimensions (height = 28px, width = 28px , canal = 1)
train_image =np.array(train).reshape(-1,28,28,1)
test100_image =np.array(test100).reshape(-1,28,28,1)
# Encodage du label de l'ensemble d'apprentissage
from keras.utils.np_utils import to_categorical

train_label = to_categorical(train_label)
test100_label = to_categorical(test100_label)
# import de Keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Convolutional Neural Network (CNN)

# Un réseau en couches séquentielles
classifier = Sequential()
# Premier étage de convolution - pooling
classifier.add(
    Conv2D(32,                      # Taille du filtre
           (3, 3),                  # Taille de la fenêtre
           padding = 'Same',        # Remplissage des bords externes
           activation="relu",       # Fonction d'activation
           input_shape=(28, 28, 1)  # Dimension de l'entrée (uniquement pour la 1ère couche)
          )
)

classifier.add(
    MaxPooling2D(
        pool_size = (2, 2)
    )
)

# Deuxième niveau 
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Applanissement (étranglement)
classifier.add(Flatten())
# Couche totalement connectée 
classifier.add(Dense(units = 256, activation = 'relu'))

# Couche de sortie (nos 10 classes-chiffres)
classifier.add(Dense(units = 10, activation = 'softmax'))
# Compilation du CNN décrit
classifier.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
# Entrainement du Réseau
epochs= 5
batch_size=90

classifier.fit(train_image, train_label, batch_size=batch_size, epochs=epochs)
# Evaluation des résultats sur les données de test
results = classifier.predict(test100_image)
results
# Sauvegarde des résultats
pred = []
numTest = results.shape[0]
# Pour chacun des items de test
for i in range(numTest):
    # Neurone de sortie de plus grande valeur
    pred.append(np.argmax(results[i])) 
predictions = np.array(pred) 

sample_submission = pd.read_csv('../input/test-100-premiers/sample_submission100.csv')
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':predictions})
result.to_csv('submission.csv',index=False)
print(result)