import pandas as pd   # lire et manipuler les IO .csv
#import numpy as np   
#import matplotlib 
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split #découper l'ensemble de train et de test de maniere aléatoire
#import math
#df=pd.read_csv('/home/notebooks/Amine/Bank DataSets Experimentations/Give me some credit.csv', sep=',')#lire le fichier csv 
df=pd.read_csv("../input/Give me some credit.csv", sep=',')
df.head()  # afficher les 5 premieres lignes
df.info() #donner les infos de notre data frame
#df['SeriousDlqin2yrs'].unique()
df.columns # citer les colonnes
# définir les attraibuts qui nous intéréssent, ici j'ai éliminé les attributs qui contiennent des nulles
df_features = df[['ID', 'RevolvingUtilizationOfUnsecuredLines', 'Age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', #'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       #'NumberOfDependents'
                 ]]
# définir l'attribut classe
df_labels = df[['SeriousDlqin2yrs']]

#import seaborn as sns
# schématiser la distribution des classes


#sns.set(style="darkgrid")
#ax = sns.countplot(  df["SeriousDlqin2yrs"])
#ax =countplotss(df["SeriousDlqin2yrs"])
#%matplotlib inline
from sklearn.model_selection import train_test_split
#decouper le data set en 30% pour test et 70% pour train
X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.3, random_state=42)
print('x_train shape:', X_train.shape) # .shape permet de voir la
print('x_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

############# Machine Learning approaches
batch_size = 32#32
epochs = 15

#from vpython import *

from keras.callbacks import ModelCheckpoint

# pour sauvegarder et màj à chaque epoch notre modèle : 
#checkpoint = [ModelCheckpoint(filepath='modNN_.hdf5')]

# importer à partir du fichier visual_callbacks.py la fonction ConfusionMatrixPlotter


#Cellule de declaration de fonctions de visualisation à chaque fin d'époch

from keras.callbacks import Callback
import matplotlib.pyplot as plt    
import matplotlib.patches as mpatches  
from sklearn.metrics import confusion_matrix
import itertools
import time
from time import sleep
import numpy as np


class AccLossPlotter(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph. 
    The graph is updated by the 'on_epoch_end' event of the Keras Callback class
    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event 
    """

    def __init__(self, graphs=['acc', 'loss'], save_graph=False):
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
        self.cmap = cmap
        plt.ion()
        #plt.show()
        plt.figure()

        plt.title(self.title)
        
        

    def on_train_begin(self, logs={}):
        pass

    
    def on_epoch_end(self, epoch, logs={}):    
        plt.clf()
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)
   
        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        thresh = cnf_mat.max() / 2.
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

#from visual_callbacks import ConfusionMatrixPlotter
#from visual_callbacks import AccLossPlotter
#plotter : changement d'accuracy et du loss à chaque epoch
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
# plotter1 : matrice de confusion à chaque epoch
class_names = ['0', '1']
plotter1 = ConfusionMatrixPlotter(X_val=X_test, classes=class_names, Y_val=y_test)

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to preventoverfitting
import keras
from keras.models import Sequential

#import sys
#import numpy as np
import tensorflow as tf
#from datetime import datetime
from keras.layers import Dense, Activation, Flatten, Dropout


model = Sequential()
model.add(Dense(128, input_shape=(9,)))#max_words, X_train.shape[1:]
model.add(Activation('relu'))
#model.add(Dense(256 ))
#model.add(Activation('relu'))
model.add(Dense(64 ))
model.add(Activation('relu'))
model.add(Dropout(0.5))
    #model.add(Dense(num_classes,))#num_classes
    #model.add(Activation('softmax'))
model.add(Dense(1, activation='sigmoid'))#
    
#model.compile(loss='binary_crossentropy',#sparse_categorical_crossentropy
#              optimizer='adam',
#              metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(X_train, y_train, nb_epoch=30, batch_size=256, callbacks=[plotter])

score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)

print('Test accuracy:', score[1])

