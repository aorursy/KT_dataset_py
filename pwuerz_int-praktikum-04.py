import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # accessing directory structure

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
import matplotlib.pyplot as plt # plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

train = pd.read_csv("../input/inp04/train.csv")
test = pd.read_csv("../input/inp04/test.csv")

#print(train_data)
# matlab mit train Daten
for i in range(len(train)):
    point = train.iloc[i]
    color = "red"
    if point[2] == 0:
        color = "blue"
    plt.scatter(point[0], point[1], s=20, c=color)
    
train.head()
#Dataframe Aufteilen (Koordinaten in X , Class in Y)
X_test = test[["X","Y"]].values
Y_test = test["class"].values

# Trainings- und Testdaten erzeugen
X_train, X_test, Y_train, Y_test = train_test_split(X_test, Y_test, random_state=0, test_size = 0.2)
#KNN Klassifitierung mit k=1. Danach auf die TrainDaten anwenden. Hilfsfunktion = fit
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)

#Test führt fast immer zu 100%. Hilffunktion : score
print('Accuracy KNN training set: {:.1f}'
     .format(knn.score(X_train, Y_train)))
print('Accuracy KNN test set: {:.1f}'
     .format(knn.score(X_test, Y_test)))
# TestSet preparieren. Danach mit den X und Y Koordinaten des TestSet's eine Vohrherrsage machen.

TestSet = test[["X","Y"]].values
TestC = test['class'].values
knn.predict(TestSet)

# prediction in ein DataFrame Speichern
prediction = pd.DataFrame()
id = []
for i in range(len(TestSet)):
    id.append(i)
    i = i + 1
    
# Struktur der Ausgabe
prediction["Id (String)"] = id 
prediction["Class (String)"] = knn.predict(TestSet).astype(int)
print(prediction)
prediction.to_csv("predictKNN.csv", index=False)
# matlab mit test Daten
for i in range(len(test)):
    point = train.iloc[i]
    color = "red"
    if point[2] == 0:
        color = "blue"
    plt.scatter(point[0], point[1], s=20, c=color)
    
# Diese Testdaten erklären die 100% ;) . Selten bekamm ich auch mal Werte leicht unter 100%.
#print(prediction)

predictionC = prediction["Class (String)"].values
#print(predictionC)

# Funktion, die die Genauigkeit der Vorhersage berechnet
def calculate_accuracy(pclass, target):
    counter = 0

    for i in range(len(pclass)):
        if pclass[i] == target[i]:
            counter += 1

    return counter / len(pclass)


print(calculate_accuracy(predictionC, TestC))