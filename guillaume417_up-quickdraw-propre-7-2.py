import datetime as time
import pandas as pd
import numpy as np
import os
import csv
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.metrics import categorical_accuracy,top_k_categorical_accuracy
from keras.models import load_model
from keras.applications import MobileNet
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y,3)


#Dossier = '../input/csv-32/'
Dossier = '../input/csv-64/'

TailleImage = 64

NbCategorie = 340
NbCSV = 100
TaillePacket = 680

NbCSVTrain = int(NbCSV*(9/10))
NbCSVValid = NbCSV-NbCSVTrain

NbEtapesTrain = 1000
NbEtapesValid = NbEtapesTrain/10
Epoch = 30

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def InitialisationPacketImage(TaillePacket, TailleImage):
    return np.zeros((TaillePacket, TailleImage, TailleImage, 1))
def InitialiserImage(Taille):#Création d'une liste composé de n sous liste de taille n
    return [[0 for j in range(Taille)] for i in range(Taille)]
def TrouverMax(Image,NumListe):#Trouver Le Maximum d'une liste
    Max=0
    for i in Image:
        for j in i[NumListe]:
            if j>Max:
                Max=j
    return Max
def AjouterPoint(Liste,X,Y,Valeur,TailleX,TailleY,TailleTab):#Ajoute la présence d'un trait dans la grille
    Liste[int(Y*TailleTab/TailleY)][int(X*TailleTab/TailleX)]=Valeur
def DessinerLigne(Liste,X1,Y1,X2,Y2,Valeur,TailleX,TailleY,TailleTab):
    #Ajoute un la ligne d'un vecteur dans la grille
    DifX=X2-X1
    DifY=Y2-Y1
    Ajout=0
    while Ajout<1:
        AjouterPoint(Liste,int(X1+DifX*Ajout),int(Y1+DifY*Ajout),
                           Valeur,TailleX,TailleY,TailleTab)
        Ajout=Ajout+1/TailleTab
def DessinerImage(Image,TailleTab):
    #Ajoute une image dans la grille
    Liste = InitialiserImage(TailleTab)
    NbVecteur=len(Image)
    j=0
    TailleX=TrouverMax(Image,0)+1 #+1 Pour eviter plus tard de prendre un élément
    TailleY=TrouverMax(Image,1)+1 # en dehors du tableau
    for Vecteur in Image:
        LongueurVec=len(Vecteur[0])
        i=0
        while i<LongueurVec-1:
            DessinerLigne(Liste,Vecteur[0][i],Vecteur[1][i],
                                      Vecteur[0][i+1],Vecteur[1][i+1],
                                      1,TailleX,TailleY,TailleTab)
            i=i+1
        j=j+1
    return Liste
def CréerTableauImages(TableauVecteur,TailleTab):
    #Réadaptation d'une base de donnée
    Liste=[]
    for Image in TableauVecteur:
        Liste.append(DessinerImage(Image,TailleTab))
    Liste = np.asarray(Liste).reshape(len(Liste), TailleTab, TailleTab)
    return np.asarray(Liste).reshape(len(Liste), TailleTab, TailleTab,1)

def PredictionCategorie(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])
def Generateur_Pack_Image(TailleImage, TaillePacket, NbCSV, NbCSVTrain, NbCSVValid, Type):
    ModifK = 0
    ModifK2 = 0
    if(Type == 'Valid'):
        ModifK = NbCSVTrain
    else:
        ModifK2 = NbCSVValid
    while True:
        for k in np.random.permutation(NbCSV-ModifK-ModifK2):
            NomFichier = os.path.join(Dossier, 'Entrainement_k{}.csv.gz'.format(ModifK+k))
            for DataFrame in pd.read_csv(NomFichier, chunksize=TaillePacket):
                DataFrame['drawing'] = DataFrame['drawing'].apply(json.loads)
                Images = InitialisationPacketImage(len(DataFrame),TailleImage)

                for i, image in enumerate(DataFrame.drawing.values):
                    Images[i, :, :, 0] = image

                Solutions = to_categorical(DataFrame.Solution, num_classes=NbCategorie)
                yield Images, Solutions
generator = Generateur_Pack_Image(TailleImage, TaillePacket, NbCSV, NbCSVTrain, NbCSVValid, 'Train')

validation_dataset = Generateur_Pack_Image(TailleImage, TaillePacket, NbCSV, NbCSVTrain, NbCSVValid, 'Valid')
    
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (TailleImage,TailleImage,1)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048, activation = "relu"))
model.add(Dropout(0.45))

model.add(Dense(NbCategorie, activation='softmax'))
model.summary()
reduction_apprentissage = ReduceLROnPlateau(monitor='val_loss', factor=0.65,
                              patience=1, verbose =1)

arret_premature = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=2) 

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics=["accuracy", top_3_accuracy])

hists = []

hist = model.fit_generator(generator,
                           steps_per_epoch= NbEtapesTrain,
                            validation_data=validation_dataset,
                            validation_steps=NbEtapesValid,
                            epochs=Epoch, verbose=1, callbacks=[reduction_apprentissage,arret_premature])

hists.append(hist)
#Diagramme Apprentissage
hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
hist_df.index = np.arange(1, len(hist_df)+1)
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
axs[0].plot(hist_df.val_acc, lw=5, label='Validation Accuracy')
axs[0].plot(hist_df.acc, lw=5, label='Training Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].grid()
axs[0].legend(loc=0)
axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')
axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')
axs[1].set_ylabel('MLogLoss')
axs[1].set_xlabel('Epoch')
axs[1].grid()
axs[1].legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show();
Chemin = '../input/quickdraw-doodle-recognition/'
Fichiers = os.listdir(os.path.join(Chemin,'train_simplified'))

ListeNomFichier = {i: v[:-4].replace(' ', '_') for i, v in enumerate(Fichiers)} 

Test = pd.read_csv(os.path.join(Chemin, 'test_simplified.csv'))
Test.head()
Test['drawing']= Test['drawing'].apply(json.loads)
#print(ListeNomFichier)

x_Test = CréerTableauImages(Test.drawing.values,TailleImage)
print(x_Test.shape)
Prediction = model.predict(x_Test, batch_size=128, verbose=1)
top3prediction = PredictionCategorie(Prediction)
top3prediction.head()
top3prediction = top3prediction.replace(ListeNomFichier)
top3prediction.head()

Test['word'] = top3prediction['a'] + ' ' + top3prediction['b'] + ' ' + top3prediction['c']
submission = Test[['key_id', 'word']]
submission.to_csv('submission.csv', index=False)
submission.head()
submission.shape
