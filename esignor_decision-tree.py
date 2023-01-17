# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot

from sklearn import tree # decision tree

from sklearn.tree import DecisionTreeClassifier # criteri

from sklearn.metrics import accuracy_score # per calcolare l'accuratezza delle previsioni

import graphviz  # permette di visualizzare l'albero di decisione



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


iris = pd.read_csv('../input/iris/Iris.csv', nrows=150, usecols=[1,2,3,4,5], encoding='latin-1') # creazione del dataframe

iris.head(10) # stampa delle prime 10 righe del dataframe



# Esportazione di un DT in formato DOT

def DT_attribute(decision_tree):

    data_DT = tree.export_graphviz(decision_tree, out_file=None, 

    feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],# nome delle caratteristiche  

    class_names=['setosa', 'versicolor', 'virginica'],# nome delle classi target  

    filled=True,# dipinge i nodi per indicare la classe di maggioranza per la classificazione 

    rounded=True,# fa i riquadri dei nodi con angoli arrotondati

    # proportion = True,# presenta i "valori" e / o "campioni" in proporzioni e percentuali

    special_characters=True) # i caratteri speciali sono per compatibilità PostScript

    graph_DT = graphviz.Source(data_DT)  

    return graph_DT # stampa del DT
decision_tree = tree.DecisionTreeClassifier() # inizializzo l'albero di decisione



sample = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] # inizializzo con il totale dei campioni

label = iris['Species'] # inizializzo con il totale delle etichette



sample_train = sample[0:120] # dati di test sui campioni

label_train = label[0:120] # dati di test sulle etichette

sample_test = sample[120:150]# dati di validazione sui campioni

label_test = label[120:150] # dati di validazione sulle etichette
decision_tree = DecisionTreeClassifier(criterion = 'entropy').fit(sample_train,label_train) # applico il criterio di entropia durante la costruzione dell'albero



prediction1 = decision_tree.predict(sample_test) # effettuo previsione sui campioni di test



print("La previsione ha un'accuratezza di: ", round(accuracy_score(label_test,prediction1)* 100), "%") # calcolo la precisione della previsione con entropia

DT_attribute(decision_tree) # stampa dell'albero di decisione sui dati di training
decision_tree = DecisionTreeClassifier(criterion = 'gini').fit(sample_train,label_train) # applico il criterio di gini durante la costruzione dell'albero



prediction2 = decision_tree.predict(sample_test) # effettuo previsione sui campioni di test



print("La previsione ha un'accuratezza di: ", round(accuracy_score(label_test,prediction2)* 100), "%") # calcolo la precisione della previsione con gini

DT_attribute(decision_tree) # stampa dell'albero di decisione sui dati di training
# disegno plot



plt.rcParams['figure.figsize'] = [15, 10] # ridimensiono l'area di stampa del plot



plt.xlabel('Id iris')

plt.ylabel('Specie')

plt.plot(range (121,151), prediction1, 'o', color='orange')

plt.plot(range (121,151), prediction2, '.', color='purple')

plt.plot(range (121,151), label_test, 'r--', color='red')

plt.legend(['entropy','gini','label esatte'], numpoints=1)

plt.show()