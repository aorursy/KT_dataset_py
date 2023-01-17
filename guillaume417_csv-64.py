import json
import os
import datetime as time
from tqdm import tqdm
import pandas as pd
import numpy as np

NombreDeCSV = 100
NombreImageParFichier = 10000

NombreImageParCSV = NombreImageParFichier//NombreDeCSV

TailleImage = 64
Dossier = os.listdir('../input/train_simplified/')

ListeNomFichier = {i: v[:-4] for i, v in enumerate(Dossier)} 

NombreFichier = len(ListeNomFichier)
print(NombreFichier)
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
    return Liste
Debut = time.datetime.now()

for i, categorie in tqdm(enumerate(ListeNomFichier)):
     for k in range(NombreDeCSV):
        NomDeFichier = 'Entrainement_k{}.csv'.format(k)
        #print(ListeNomFichier[categorie])
        DataFrame = pd.read_csv(os.path.join('../input/train_simplified', ListeNomFichier[categorie] + '.csv'),
                         skiprows=range(1,NombreImageParCSV*k), 
                         nrows=NombreImageParCSV, 
                         usecols=['word','drawing'])
        
        DataFrame['Solution'] = i        
        #print(df["word"]+ " + " +df['drawing'])
        if i == 0:
            DataFrame.to_csv(NomDeFichier, index=False)
        else:
            DataFrame.to_csv(NomDeFichier, mode='a', header=False, index=False)
for k in tqdm(range(NombreDeCSV)):
    NomDeFichier = 'Entrainement_k{}.csv'.format(k)
    if os.path.exists(NomDeFichier):
        #Melange
        DataFrame = pd.read_csv(NomDeFichier)
        DataFrame['rnd'] = np.random.rand(len(DataFrame))
        DataFrame = DataFrame.sort_values(by='rnd').drop('rnd', axis=1)
        #Vecteur -> Image
        DataFrame['drawing'] = DataFrame['drawing'].apply(json.loads)
        DataFrame['drawing'] = CréerTableauImages(DataFrame.drawing.values, TailleImage)
        DataFrame['drawing'] = DataFrame['drawing'].apply(json.dumps)
          
        DataFrame.to_csv(NomDeFichier + '.gz', compression='gzip', index=False)
        os.remove(NomDeFichier)
        
print(DataFrame.shape)
print(DataFrame)
Fin = time.datetime.now()
print('Dernier Lancement {}.\nTemps Total {}s'.format(Fin, (Fin - Debut).seconds))