# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import datasets

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import seaborn as sns
#Questo dataset è composto da 1797 immagini da 8×8 pixel ognuna.

#Ogni immagine rappresenta una cifra scritta a mano.

data = datasets.load_digits()



#RAPPRESENTAZIONE DEL DATASET:

#immagine che raffigura 8 righe e 8 colonne di numeri scritti a mano.

#Il valore in basso a sinistra indica l’etichetta che 

#il valore scritto a mano rappresenta.



fig = plt.figure(figsize=(8,8))

for i in range(64):

    x = fig.add_subplot(8,8,i+1,xticks = [], yticks = [])

    x.imshow(data.images[i], cmap = plt.cm.binary, interpolation = 'nearest')

    x.text(0,7, str(data.target[i]))
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size= 0.3)
#Funzione che permette di visualizzare il risultato del classificatore

#in cui l’etichetta in basso a sinistra verde indicherà una previsione buona del modello e

#l’etichetta in basso a sinistra rossa indicherà una previsione errata del modello.



def plot(fig, predicted, expected):

    for i in range(64):

        x = fig.add_subplot(8,8,i+1,xticks = [], yticks = [])

        x.imshow(X_test.reshape(-1,8,8)[i], cmap = plt.cm.binary, interpolation = 'nearest')

        if predicted[i] == expected[i]:

            x.text(0,7,str(predicted[i]), color = 'green')

        else:

            x.text(0,7,str(predicted[i]), color = 'red')



#Funzione che permette di visualizzare la matrice di confusione 

#e il report di classificazione. 

#Matrice di confusione: 

#la diagonale contiene le previsioni corrette;

#gli errori di previsione saranno rappresentati dai valori esterni alla diagonale.

#Report di classificazione:

#Precision: per ogni classe è definito come il rapporto 

#tra veri positivi e la somma di veri e falsi positivi. 

#Recall: Per ogni classe è definito come il rapporto tra 

#i veri positivi e la somma dei veri positivi e dei falsi negativi.

#F-score: è una media armonica ponderata delle metriche Precision e Recall 

#in modo tale che il punteggio migliore sia 1 e il peggiore sia 0. 

#Accuracy: indica l’accuratezza del modello.

#Macro avg: media della media non ponderata per etichetta.

#Weighted avg: media della media ponderata per etichetta.



def plot_CM_CR(expected, predicted):

    cm = metrics.confusion_matrix(expected, predicted)

    print("CONFUSION MATRIX")

    sns.heatmap(cm,annot=True,fmt='.0f')

    plt.show()

    print(metrics.classification_report(expected, predicted))
#Il modello di Naive Bayes multinomiale è utilizzato in genere 

#per la classificazione dei documenti e si basa su una distribuzione 

#di probabilità multinomiale. 



model = MultinomialNB()

model.fit(X_train, y_train)

predicted_NB = model.predict(X_test)

expected_NB = y_test

fig = plt.figure(figsize=(8,8))

plot(fig, predicted_NB, expected_NB)
plot_CM_CR(expected_NB, predicted_NB)
#Random Forest rappresenta un tipo di modello ensemble, 

#che si avvale del bagging come metodo di ensemble

#e l’albero decisionale come modello individuale.

#Il risultato finale nel caso della classificazione 

#è la classe restituita dal maggior numero di alberi 



rfc = RandomForestClassifier(n_estimators = 100)

model = rfc.fit(X_train, y_train)

predicted_RF = model.predict(X_test)

expected_RF = y_test

fig = plt.figure(figsize=(8,8))

plot(fig, predicted_RF, expected_RF)
plot_CM_CR(expected_RF, predicted_RF)