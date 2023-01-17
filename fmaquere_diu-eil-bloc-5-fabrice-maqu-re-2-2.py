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
#   par exemple le label '2' -> [ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
from keras.utils.np_utils import to_categorical
train_label_cat = to_categorical(train_label)
test100_label_cat = to_categorical(test100_label)
# import de Keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import UpSampling2D
from keras import backend as K
from keras.callbacks import Callback
import matplotlib.patches as mpatches

class AccLossPlotter(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph. 
    The graph is updated by the 'on_epoch_end' event of the Keras Callback class
    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event 
    """
    
    def __init__ (self, graphs=['acc','loss'], save_graph=False):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph
    
    
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.ion()
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]

    def on_train_end(self, epoch, logs={}):
        self.epoch_count += 1    
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0
        
        if 'loss' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Loss')
            #plt.axis([0,100,0,5])
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)
        
        plt.draw()
        plt.pause(0.001)
        
        if self.save_graph:
            plt.savefig('training_acc_loss.png')
            

# plotter : accuracy et loss à chaque cycle
plotterAccLoss = AccLossPlotter(graphs=['loss'], save_graph=True)

# Convolutional Autoencoder (CAE)

# Un réseau en couches séquentielles
cae = Sequential()
# Premier étage de convolution - pooling
cae.add(Conv2D(32,(3, 3),padding = 'Same',activation="relu",input_shape=(28, 28, 1)))
cae.add(MaxPooling2D(pool_size = (2, 2)))
cae.add(Conv2D(32, (3, 3), activation="relu", padding='same'))
cae.add(MaxPooling2D(pool_size = (2, 2)))

# Etranglement
cae.add(Dense(units = 25, activation = 'relu'))

# Premier étage de déconvolution - sampling
cae.add(Conv2D(32, (3, 3), activation="relu", padding='same'))
cae.add(UpSampling2D((2, 2)))
# Deuxième niveau 
cae.add(Conv2D(32, (3, 3), activation="relu", padding='same'))
cae.add(UpSampling2D((2, 2)))

# Couche de sortie
cae.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compilation du CNN décrit
cae.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

# Sauvegarde de la représentation graphique du réseau
#     Ce fichier sera accessible dans l'onglet 'output' du kernel après un commit
from keras.utils import plot_model
plot_model(cae, to_file='cae.png', show_shapes=True)


# Entrainement du Réseau
epochs= 20        # 20
batch_size=128
cae.fit(train_image, train_image,
               batch_size=batch_size,
               validation_data=(test100_image, test100_image),
               epochs=epochs, 
               callbacks=[plotterAccLoss])
# Evaluation des résultats sur les données de test
results = cae.predict(test100_image)

# Dimension de l'affichage
nb_cols = 10
nb_ligs = 2
taille  = 20

# Affichage de nb_ligs lignes de nb_cols images
fig1, ax1 = plt.subplots(2*nb_ligs,nb_cols, figsize=(taille,10))

for no_lig in range(nb_ligs):
    for no_col in range(nb_cols):
        ax1[2*no_lig][no_col].imshow(test100_image[no_lig*nb_cols+no_col].reshape((28,28)), cmap='gray')
        ax1[2*no_lig+1][no_col].imshow(results[no_lig*nb_cols+no_col].reshape((28,28)), cmap='gray')
        ax1[2*no_lig][no_col].axis('off') 
        ax1[2*no_lig+1][no_col].axis('off')
# Création des images bruitées
noise_factor = 0.5
test100_image_noisy = test100_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test100_image.shape) 
test100_image_noisy = np.clip(test100_image_noisy, 0., 1.)

# Evaluation des résultats sur les données de test
results = cae.predict(test100_image_noisy)

# Dimension de l'affichage
nb_cols = 10
nb_ligs = 2
taille  = 20

# Affichage de nb_ligs lignes de nb_cols images
fig1, ax1 = plt.subplots(2*nb_ligs,nb_cols, figsize=(taille,10))

for no_lig in range(nb_ligs):
    for no_col in range(nb_cols):
        ax1[2*no_lig][no_col].imshow(test100_image_noisy[no_lig*nb_cols+no_col].reshape((28,28)), cmap='gray')
        ax1[2*no_lig+1][no_col].imshow(results[no_lig*nb_cols+no_col].reshape((28,28)), cmap='gray')
        ax1[2*no_lig][no_col].axis('off') 
        ax1[2*no_lig+1][no_col].axis('off')