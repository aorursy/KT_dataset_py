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
testperso = pd.read_csv('../input/testpersonnel/test_perso.csv')
train_2800 = pd.read_csv('../input/testpersonnel/train_2800.csv')
# Affichage des informations
print("Il y a {0} exemples d'apprentissage.".format(train.shape[0]))
print("Il y a {0} exemples d'apprentissage.".format(train_2800.shape[0]))
print("Il y a {0} exemples d'évaluation.".format(evaluation.shape[0]))
print("Il y a {0} exemples de test.".format(test100.shape[0]))
print("Il y a {0} exemples de test personnel.".format(testperso.shape[0]))
print("Nombre de colonnes : " , train.shape[1])
print("Liste des colonnes :")
train.columns
# On sépare la classe (label) des données (pixels)
# Le vecteur des numéros de classe
train_label = train["label"]
train_2800_label = train_2800["label"]
# Suppression de cette colonne 
train.drop("label", axis = 1 , inplace=True)
train_2800.drop("label", axis = 1 , inplace=True)
# Nombre d'exemples d'entrainement par classe
train_label.value_counts().sort_index()
train_2800_label.value_counts().sort_index()
# La même chose pour les tests
test100_label = test100["label"]
test100.drop("label", axis = 1 , inplace=True)
test100_label.value_counts().sort_index()
testperso_label = testperso["label"]
testperso.drop("label", axis = 1 , inplace=True)
testperso_label.value_counts().sort_index()
# Normalisation des valeurs des images [0-255] -> [0.0-1.0]
train      = train/255.0
train_2800 = train_2800/255.0
test100    = test100/255.0
evaluation = evaluation/255.0
testperso  = testperso/255.0
# Affichage des images de l'ensemble de test

import matplotlib.pyplot as plt
%matplotlib inline

# Dimension de l'affichage
nb_cols = 10
nb_ligs = 2
taille  = 20

# Affichage de nb_ligs lignes de nb_cols images
fig1, ax1 = plt.subplots(nb_ligs,nb_cols, figsize=(taille,7))
for no_lig in range(nb_ligs):
    for no_col in range(nb_cols):
        ax1[no_lig][no_col].imshow(testperso.iloc[no_lig*nb_cols+no_col].values.reshape((28,28)), cmap='gray')
        ax1[no_lig][no_col].axis('off')
        ax1[no_lig][no_col].set_title(testperso_label[no_lig*nb_cols+no_col])  
import numpy as np
# Restructuration des images en 3 dimensions (height = 28px, width = 28px , canal = 1)
train_image =np.array(train).reshape(-1,28,28,1)
train_2800_image =np.array(train_2800).reshape(-1,28,28,1)
test100_image =np.array(test100).reshape(-1,28,28,1)
testperso_image =np.array(testperso).reshape(-1,28,28,1)
# Restructuration des images dans un tensor 2D (nb = 2800, pixels = 28x28) pour un réseau simple sans convolution
train_2800_NN_image = np.array(train_2800).reshape((2800, 28 * 28))
test100_NN_image = np.array(test100).reshape((100, 28 * 28))
testperso_NN_image = np.array(testperso).reshape((20, 28 * 28))
# Encodage du label de l'ensemble d'apprentissage
#   par exemple le label '2' -> [ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
from keras.utils.np_utils import to_categorical

train_label_cat = to_categorical(train_label)
train_2800_label_cat = to_categorical(train_2800_label)
test100_label_cat = to_categorical(test100_label)
testperso_label_cat = to_categorical(testperso_label)
# import de Keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
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

    def on_train_end(self, epoch, logs={}):
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

    def on_train_end(self, epoch, logs={}):        
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
# Convolutional Neural Network (CNN)

# Un réseau en couches séquentielles
classifier = Sequential()
# Premier étage de convolution - pooling
classifier.add(Conv2D(32,(3, 3),padding = 'Same',activation="relu",input_shape=(28, 28, 1)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Deuxième niveau 
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Applanissement (étranglement)
classifier.add(Flatten())
# Couche totalement connectée 
classifier.add(Dense(units = 512, activation = 'relu'))
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
plot_model(classifier, to_file='cnn1.png', show_shapes=True)

# Entrainement du Réseau
epochs= 30        # 20
batch_size=90
classifier.fit(train_image, train_label_cat,
               batch_size=batch_size,
               validation_data=(test100_image, test100_label_cat),
               epochs=epochs, 
               callbacks=[plotterAccLoss,plotterCM])
# Evaluation des résultats sur les données de test
results = classifier.predict(testperso_image)
# results
# Sauvegarde des résultats
pred = []
numTest = results.shape[0]
# Pour chacun des items de test
for i in range(numTest):
    # Neurone de sortie de plus grande valeur
    pred.append(np.argmax(results[i])) 
predictions = np.array(pred) 
sample_submission = pd.read_csv('../input/testpersonnel/sample_submissionperso.csv')
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label prédit':predictions, 'Classe réelle':testperso_label})
result.to_csv('submission.csv',index=False)
print(result)
# Convolutional Neural Network (CNN)

# Un réseau en couches séquentielles
cnn2800 = Sequential()
# Premier étage de convolution - pooling
cnn2800.add(Conv2D(32,(3, 3),padding = 'Same',activation="relu",input_shape=(28, 28, 1)))
cnn2800.add(MaxPooling2D(pool_size = (2, 2)))
# Deuxième niveau 
cnn2800.add(Conv2D(32, (3, 3), activation="relu"))
cnn2800.add(MaxPooling2D(pool_size = (2, 2)))
# Applanissement (étranglement)
cnn2800.add(Flatten())
# Couche totalement connectée 
cnn2800.add(Dense(units = 512, activation = 'relu'))
# Couche de sortie (nos 10 classes-chiffres)
cnn2800.add(Dense(units = 10, activation = 'softmax'))
# Compilation du CNN décrit
cnn2800.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

# Sauvegarde de la représentation graphique du réseau
#     Ce fichier sera accessible dans l'onglet 'output' du kernel après un commit
from keras.utils import plot_model
plot_model(cnn2800, to_file='cnn2800.png', show_shapes=True)

# Entrainement du Réseau
epochs= 10        # 20
batch_size=90
cnn2800.fit(train_2800_image, train_2800_label_cat,
               batch_size=batch_size,
               validation_data=(test100_image, test100_label_cat),
               epochs=epochs, 
               callbacks=[plotterAccLoss])
# Neural Network (NN)

# Un réseau en couches séquentielles
nn2800 = Sequential()
# 1 couche totalement connectée
nn2800.add(Dense(units = 512, activation = 'relu', input_shape=(28*28,)))
# Couche de sortie (nos 10 classes-chiffres)
nn2800.add(Dense(units = 10, activation = 'softmax'))
# Compilation du CNN décrit
nn2800.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

# Sauvegarde de la représentation graphique du réseau
#     Ce fichier sera accessible dans l'onglet 'output' du kernel après un commit
from keras.utils import plot_model
plot_model(nn2800, to_file='nn2800.png', show_shapes=True)

# Entrainement du Réseau
epochs= 10       # 20
batch_size=90
nn2800.fit(train_2800_NN_image, train_2800_label_cat,
               batch_size=batch_size,
               validation_data=(test100_NN_image, test100_label_cat),
               epochs=epochs, 
               callbacks=[plotterAccLoss])
# Evaluation des résultats sur les données de test
results = cnn2800.predict(test100_image)
resultsNN = nn2800.predict(test100_NN_image)
# results
# Sauvegarde des résultats
pred = []
predNN = []
numTest = results.shape[0]
# Pour chacun des items de test
for i in range(numTest):
    # Neurone de sortie de plus grande valeur
    pred.append(np.argmax(results[i])) 
    predNN.append(np.argmax(resultsNN[i]))
predictions = np.array(pred) 
predictionsNN = np.array(predNN) 
sample_submission = pd.read_csv('../input/test-100-premiers/sample_submission100.csv')
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Classe réelle':test100_label, 'Label prédit CNN':predictions, 'Label prédit NN':predictionsNN})
result.to_csv('submission.csv',index=False)
#print(result)
compt = 0
comptNN = 0
for i in range(0,99):
    if test100_label[i] != predictions[i]:
        compt+=1
    if test100_label[i] != predictionsNN[i]:
        comptNN+=1
print("Comparaison des 2 réseaux")
print("   Réseau CNN : {0} erreurs" .format(compt))
print("   Réseau NN : {0} erreurs" .format(comptNN))
cnn_loss, cnn_acc = cnn2800.evaluate(test100_image, test100_label_cat)
nn_loss, nn_acc = nn2800.evaluate(test100_NN_image, test100_label_cat)
print("Loss")
print("   Réseau CNN : {:.2f}" .format(cnn_loss))
print("   Réseau NN : {:.2f}" .format(nn_loss))
print("Accuracy")
print("   Réseau CNN : {:.2f}" .format(cnn_acc))
print("   Réseau NN : {:.2f}" .format(nn_acc))