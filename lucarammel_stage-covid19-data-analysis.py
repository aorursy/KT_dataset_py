# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Lecture du dataset et prétraitement



t = float(input()) #Temps en minutes d'un cycle. 



data = pd.read_excel('/kaggle/input/lp02-bistable-data/LP02-1.xlsx')

data = data.drop(['Unnamed: 0'],axis = 1)

data.rename(columns={'Cycle':'Time'}, inplace = True)

data.Time = data.index*t

data_lesstime = data.drop(['Time'],axis = 1)



data
#Paramètres d'affichages des expériences à définir



n = np.shape(data_lesstime)



print("Rentrer le nombre d'expériences")

nexp = int(input()) #Nombre d'expériences 



print('Rentrer le nombre de points par expériences')

nrange = int(input()) #Nombre de points sur la range d'un paramètre de l'expérience



print('Rentrer les différents titres des expériences')

titles = list(map(str, input().split())) 



print('Rentrer le type de normalisation pour les courbes')

normalisation = int(input()) #0 si -min/max , 1 si -min

#Renommer les colonnes du tableau rapidement 



valeur_concentration = list(map(float, input().split()))

dictionary = {}



for i in range(len(data_lesstime.columns)):

    if i//nrange < len(titles) :

        dictionary.update({data_lesstime.columns[i] : titles[(i//nrange)]+'_{}'.format(valeur_concentration[i%nrange])})

    else : 

        dictionary.update({data_lesstime.columns[i] : titles[(len(titles))-1]+'_{}'.format(valeur_concentration[i%nrange])})

        

print(dictionary)

data_lesstime.rename(columns = dictionary, inplace = True)



names = data_lesstime.columns

#Affichage des courbes

for i in range(0,n[1]):

    quotient = i//nrange

    plt.figure(quotient)

    plt.plot(data['Time'],data_lesstime.iloc[:,i],label = names[i])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),shadow=True, ncol=5)

    plt.ylabel('Fluorescence')

    plt.xlabel("Temps (s)")



#Affichage des titres et des légendes

for j in range(0,nexp):

    plt.figure(j)

    plt.title(titles[j])

    
#Après normalisation des données : 

seuil = min((data_lesstime.max()-data_lesstime.min()))



data_scaled = pd.DataFrame(columns = names, index= range(n[0]))

data_scaled.fillna(0, inplace = True)



for i in range(n[1]):

    if normalisation == 0 : 

        if max(data_lesstime.iloc[:,i]-min(data_lesstime.iloc[:,i]))>seuil*1.5:

            data_scaled.iloc[:,i]=(data_lesstime.iloc[:,i]-min(data_lesstime.iloc[:,i]))/(max(data_lesstime.iloc[:,i]-min(data_lesstime.iloc[:,i])))

    elif normalisation == 1: 

        data_scaled.iloc[:,i]=(data_lesstime.iloc[:,i]-min(data_lesstime.iloc[:,i]))

    else : 

        data_scaled.iloc[:,i] = (data_lesstime.iloc[:,i]-min(data_lesstime.iloc[:,i]))



print("Seuil du bruit : {}".format(seuil))
#Affichage des courbes normalisées 

for i in range(0,n[1]):

    quotient = i//nrange

    plt.figure(quotient)

    plt.plot(data['Time'],data_scaled.iloc[:,i],label = names[i])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),shadow=True, ncol=5)

    plt.ylabel('Fluorescence')

    plt.xlabel("Temps (s)")





#Affichage des titres et des légendes

for j in range(0,nexp):

    plt.figure(j)

    plt.title(titles[j])

    plt.savefig('fluovstime{}.png'.format(j))

#Détermination des temps d'amplification 

Tampli = []

i = 0



for j in range(0,n[1]):

    while data_scaled.iloc[i,j]<0.4 and i!= (n[0]-1) :

        i=i+1

    Tampli.append(data.iloc[i,0])

    i=0

    

Tamp = pd.DataFrame(Tampli)        

Tamp.index = names

Tamp.columns = ["Temps d'amplification (min)"]



Tamp
#Afficher les temps en fonction de concentrations. 



cnc=Tamp.index[Tamp.index.str.contains(pat='NC')] #Selectionne les index où la chaine de caractère est NC = negative control

cpc=Tamp.index[Tamp.index.str.contains(pat='PC')]





Tamp.T[cnc].T.plot.bar()

plt.title("Temps d'amplification pour les controles négatifs")



Tamp.T[cpc].T.plot.bar()

plt.title("Temps d'amplification pour les controles positifs")
