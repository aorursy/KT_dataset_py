#Geforked von Uwe. 

import numpy as np
import pandas as pd
#Einlesen der Daten
df = pd.read_csv("../input/dataset/train.csv")
df.head()
#Hier splitten wir die Daten in Trainings und Testdaten auf
from sklearn.model_selection import train_test_split

X = df[["X","Y"]].values
Y = df["class"].values
colors = {0:'red',1:'blue'}

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.25)

#Wir plotten hier unsere Punkte und färben diese ein. So sehen wir den Aufbau und die Anordnung der Daten.
%matplotlib inline
import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],c=df["class"].apply(lambda x: colors[x]))
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#print(X_Test)
#print(Y_Test)
#Nun trainieren wir das Modell mit verschiedenen Parametern für n_neighbor, damit wir wissen, welchen Wert wir übergeben müssen, um den besten Wert zu bekommen.
#Hier ist es egal, da alles hundert Prozent liefert.
from sklearn.neighbors import KNeighborsClassifier

test_accuracy = []

neighbors_range = range(1,50)

for n_neighbors in neighbors_range:
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_Train, Y_Train)
    test_accuracy.append(clf.score(X_Test, Y_Test))    
    
plt.plot(neighbors_range, test_accuracy, label='Genauigkeit bei den Testdaten')
plt.ylabel('Genauigkeit')
plt.xlabel('Anzahl der Nachbarn')
plt.legend()
#Jetzt trainieren wir unser Modell speziell mit n_neighbor=2 und geben die Score aus. 1.0!!!!
model = KNeighborsClassifier(n_neighbors = 2)
model.fit(X_Train, Y_Train)

model.predict(X_Test)

print(model.score(X_Test,Y_Test))
#Wird für externe .py-Datein benötigt.
import importlib.machinery
modulename = importlib.machinery.SourceFileLoader('helper','../input/helperscript/helper.py').load_module()

# 1. Bild mit Trennlinie
from helper import plot_classifier

plot_classifier(model,X, Y, proba = False, xlabel = "x", ylabel="y")

# 2. Bild mit Wahrscheinlichkeiten
from helper import plot_classifier

plot_classifier(model,X, Y, proba = True, xlabel = "x", ylabel="y")
######### hier versuchen wir nun das Vorhersagen#######
#Zuerst lesen wir die Testdatei ein
testdf = pd.read_csv("../input/dataset/test.csv")
XTEST = testdf[["X","Y"]].values
YTEST = testdf["class"].values

#Wir plotten die Testdaten
%matplotlib inline
import matplotlib.pyplot as plt

plt.scatter(XTEST[:,0],XTEST[:,1],c=testdf["class"].apply(lambda x: colors[x]))
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Wir geben die Testdaten aus, um diese später mit unserer Vorhersagen zu vergleichen.
print(testdf)

testX = testdf[["X","Y"]].values
#Hier sagen wir die Werte vorher anhand unseres Modells
prediction = model.predict(testX)

#Wir geben nun die Prediction aus und können diese mit den richtigen Werten vergleichen
print(prediction)
########################ENDE#################################