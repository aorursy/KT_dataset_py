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
plot_model(classifier, to_file='model_classifier_digit.png', show_shapes=True)
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
epochs= 10        # 20
batch_size=90

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