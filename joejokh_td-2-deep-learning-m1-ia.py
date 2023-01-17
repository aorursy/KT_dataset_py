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

# Affichage des images de l'ensemble de test_100

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
from keras.layers import Dropout
from keras.layers import Activation
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
# Sauvegarde de la représentation graphique du réseau
#     Ce fichier sera accessible dans l'onglet 'output' du kernel après un commit
from keras.utils import plot_model
import matplotlib.image as mpimg

plot_model(classifier, to_file='model_classifier_digit.png', show_shapes=True)

# affichage de la représentation graphique du réseau
img=mpimg.imread('../working/model_classifier_digit.png')
fig, ax = plt.subplots(figsize=(20, 20))
imgplot = ax.imshow(img)
plt.show()
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
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.ion()
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0
        
        if 'acc' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accuracy')
            #plt.axis([0,100,0,1])
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

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

    def on_train_end(self, logs={}):
        if self.save_graph:
            plt.savefig('training_acc_loss.png')


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.metrics import confusion_matrix

import itertools

class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
    """
    def __init__(self, X_val, Y_val, classes, normalize=False, cmap=plt.cm.Blues, title='Confusion Matrix'):
        self.X_val = X_val
        self.Y_val = Y_val
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap             # Color map
        plt.ion()                    # Utilisation de pyplot en mode interactif
        #plt.show()
        plt.figure()

        plt.title(self.title)
        

    def on_train_begin(self, logs={}):
        pass

    
    def on_epoch_end(self, epoch, logs={}):
        """ A la fin d'un cycle """
        plt.clf()
        # On évalue le résultat du modèle sur toutes les entrées de X_val
        pred = self.model.predict(self.X_val)
        # On récupère l'index de la colonne de plus grande valeur pour toutes les lignes
        #     du résultat
        max_pred = np.argmax(pred, axis=1)
        # On récupère l'index de la colonne de plus haute valeur pour toutes les lignes
        max_y = np.argmax(self.Y_val, axis=1)
        # On évalue la matrice de confusion
        cnf_mat = confusion_matrix(max_y, max_pred)
        
        # S'il y a une demande de normalisation 
        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        # Seuil à la moitié du max
        thresh = cnf_mat.max() / 2.
        
        # Pour tous les éléments du produit cartésien (tous les éléments de la matrice)
        for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j],                                          
                         horizontalalignment="center",
                         color="white" if cnf_mat[i, j] > thresh else "black")

        plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.colorbar()
                                                                                                         
        plt.tight_layout()                                                    
        plt.ylabel('True label')                                              
        plt.xlabel('Predicted label')
            #plt.draw()
        plt.show()
        plt.pause(0.001)

# plotter : accuracy et loss à chaque cycle
plotterAccLoss = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
# plotter : matrice de confusion à chaque cycle
class_names = ['0', '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9']
plotterCM = ConfusionMatrixPlotter(
    X_val=train_image,
    classes=class_names,
    Y_val=train_label_cat)
# Entrainement du Réseau
epochs= 1        # 20
batch_size=128

classifier.fit(train_image, train_label_cat,
               batch_size=batch_size,
               validation_data=(test100_image, test100_label_cat),
               epochs=epochs, 
               callbacks=[plotterAccLoss,plotterCM])
# Evaluation des résultats sur les données de test
results = classifier.predict(test100_image)
# results
# Sauvegarde des résultats
pred = []
numTest = results.shape[0]
# Pour chacun des items de test
for i in range(numTest):
    # Neurone de sortie de plus grande valeur
    pred.append(np.argmax(results[i])) 
predictions = np.array(pred) 

sample_submission = pd.read_csv('../input/test-100-premiers/sample_submission100.csv')
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label prédit':predictions, 'Classe réelle':test100_label})
result.to_csv('submission.csv',index=False)
print(result)
from keras.layers.normalization import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU

# Un réseau en couches séquentielles
big_deep_net = Sequential()
# Premier étage de convolution - pooling
big_deep_net.add(
    Conv2D(48,                      # Taille du filtre
           (5, 5),                  # Taille de la fenêtre
           padding = 'Same',        # Remplissage des bords externes
           activation="linear",       # Fonction d'activation
           input_shape=(28, 28, 1)  # Dimension de l'entrée (uniquement pour la 1ère couche)
          )
)
big_deep_net.add(LeakyReLU(alpha=.001))
big_deep_net.add(BatchNormalization())
big_deep_net.add(Dropout(0.3))
big_deep_net.add(MaxPooling2D(pool_size = (2, 2)))

# Deuxième niveau 
big_deep_net.add(Conv2D(32, (4, 4), activation="linear"))
big_deep_net.add(LeakyReLU(alpha=.001))
big_deep_net.add(BatchNormalization())
big_deep_net.add(Dropout(0.3))
big_deep_net.add(MaxPooling2D(pool_size = (2, 2)))

# troisième niveau 
#big_deep_net.add(Conv2D(24, (3, 3), activation="linear"))
#big_deep_net.add(LeakyReLU(alpha=.001))
#big_deep_net.add(BatchNormalization())
#big_deep_net.add(Dropout(0.3))
#big_deep_net.add(MaxPooling2D(pool_size = (2, 2)))

# Applanissement (étranglement)
big_deep_net.add(Flatten())
# Couche totalement connectée 
big_deep_net.add(Dense(units = 256, activation = 'linear'))
big_deep_net.add(LeakyReLU(alpha=.001))

# Couche de sortie (nos 10 classes-chiffres)
big_deep_net.add(Dense(units = 10, activation = 'softmax'))

# Compilation du CNN décrit
big_deep_net.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

# Sauvegarde de la représentation graphique du réseau
from keras.utils import plot_model
import matplotlib.image as mpimg

plot_model(big_deep_net, to_file='big_deep_net.png', show_shapes=True)

# affichage de la représentation graphique du réseau
img=mpimg.imread('../working/big_deep_net.png')
fig, ax = plt.subplots(figsize=(10, 30))
imgplot = ax.imshow(img)
plt.show()

# Entrainement du Réseau
epochs= 1       # 20
batch_size=128

big_deep_net.fit(train_image, train_label_cat,
               batch_size=batch_size,
               validation_data=(test100_image, test100_label_cat),
               epochs=epochs, 
               callbacks=[plotterAccLoss,plotterCM])


# Un réseau en couches séquentielles
Auto_encoder = Sequential()
# Premier niveau de convolution - pooling
Auto_encoder.add(
    Conv2D(64,                      # Taille du filtre
           (4, 4),                  # Taille de la fenêtre
           padding = 'Same',        # Remplissage des bords externes
           activation="relu",       # Fonction d'activation
           input_shape=(28, 28, 1)  # Dimension de l'entrée (uniquement pour la 1ère couche)
          )
)
Auto_encoder.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

# Deuxième niveau de convolution - pooling
Auto_encoder.add(Conv2D(32, (3, 3), activation="relu" , padding='same'))
Auto_encoder.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

Auto_encoder.add(Flatten())
Auto_encoder.add(Reshape((7, 7, 32)))

# Premier niveau de deconvolution  - UpSampling2D
Auto_encoder.add(Conv2D(32, (3, 3), activation="relu" , padding='same'))
Auto_encoder.add(UpSampling2D((2, 2)))

# Deuxième niveau de deconvolution - UpSampling2D
Auto_encoder.add(Conv2D(64, (4, 4), activation="relu" , padding='same'))
Auto_encoder.add(UpSampling2D((2, 2)))


Auto_encoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compilation du CNN décrit
Auto_encoder.compile(
    optimizer = 'adadelta',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

# Sauvegarde de la représentation graphique du réseau
from keras.utils import plot_model
import matplotlib.image as mpimg

plot_model(Auto_encoder, to_file='Auto_encoder.png', show_shapes=True)

# affichage de la représentation graphique du réseau
img=mpimg.imread('../working/Auto_encoder.png')
fig, ax = plt.subplots(figsize=(10, 30))
imgplot = ax.imshow(img)
plt.show()


# Entrainement du Réseau
epochs= 1        # 20
batch_size=128

Auto_encoder.fit(train_image, train_image,
               batch_size=batch_size,
               validation_data=(test100_image, test100_image),
               epochs=epochs, 
               callbacks=[plotterAccLoss])
#reconstriction des images
decoded_imgs = classifier2.predict(test100_image)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(test100_image[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#NOtre 2eme auto-encodeur
# Un réseau en couches séquentielles
Auto_encoder2 = Sequential()

# Premier niveau de convolution - pooling
Auto_encoder2.add(Conv2D(64,(4, 4),padding = 'Same',activation="relu",input_shape=(28, 28, 1)))
Auto_encoder2.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

# Deuxième niveau de convolution - pooling
Auto_encoder2.add(Conv2D(32, (3, 3), activation="relu" , padding='same'))
Auto_encoder2.add(MaxPooling2D(pool_size = (2, 2), padding='same'))


# Deuxième niveau de deconvolution  - UpSampling2D
Auto_encoder2.add(Conv2D(32, (3, 3), activation="relu" , padding='same'))
Auto_encoder2.add(UpSampling2D((2, 2)))

# Troisième niveau de deconvolution - UpSampling2D
Auto_encoder2.add(Conv2D(64, (4, 4), activation="relu" , padding='same'))
Auto_encoder2.add(UpSampling2D((2, 2)))


Auto_encoder2.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compilation du CNN décrit
Auto_encoder2.compile(
    optimizer = 'adadelta',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])


plot_model(Auto_encoder2, to_file='Auto_encoder2.png', show_shapes=True)

# affichage de la représentation graphique du réseau
img=mpimg.imread('../working/Auto_encoder2.png')
fig, ax = plt.subplots(figsize=(10, 30))
imgplot = ax.imshow(img)
plt.show()

#Ajout de bruit.
noise_factor = 0.5
train_noisy = train_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_image.shape) 
test_noisy = test100_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test100_image.shape) 
train_noisy = np.clip(train_noisy, 0., 1.)
test_noisy = np.clip(test_noisy, 0., 1.)

# Entrainement du Réseau
epochs= 7        # 20
batch_size=128

Auto_encoder2.fit(train_noisy, train_image,
               batch_size=batch_size,
               validation_data=(test_noisy, test100_image),
               epochs=epochs, 
               callbacks=[plotterAccLoss])
#filtrage des images
filtred_imgs = Auto_encoder2.predict(test_noisy)

n = 9
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(filtred_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()