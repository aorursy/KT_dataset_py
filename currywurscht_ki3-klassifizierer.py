import numpy as np # linear algebra
from numpy import linalg as LA
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data
input = "../input/kiwhs-comp-1-complete/train.arff"
daten = read_data(input)
def sortiere_daten(daten):
    minusCoords = []
    plusCoords = []
    for i in daten :
        if i[2] == -1:
             minusCoords = minusCoords + i
        if i[2] == 1:    
            plusCoords = plusCoords + i  
    return minusCoords, plusCoords
negativ,positiv = sortiere_daten(daten)
def errechne_Durchschnitt(koordinaten):
    #Python will explizit ein Array fuer Gleitkommazahlen haben.
    durchschnitt = np.array([0.0,0.0])
    for i in koordinaten:
        #aufaddieren der werte
        durchschnitt += i
    durchschnitt = durchschnitt /  len(koordinaten)
    return durchschnitt
negativ = errechne_Durchschnitt(negativ)
print("\"Durchschnittliche\" Koordinate fuer -1", negativ)
positiv = errechne_Durchschnitt(positiv)
print("\"Durchschnittliche\" Koordinate fuer 1",positiv)
def klassifiziere(punkt):
    klasse = 0
    if LA.norm(punkt -positiv) < LA.norm(punkt -negativ):
        klasse = 1
    elif LA.norm(punkt - positiv) > LA.norm(punkt - negativ):
        klasse = -1
    return klasse 
#kleiner Test mit einem Punkt aus test.csv
print(klassifiziere([-2.151969,-0.530527]))
#Testdaten einlesen und submission erstellen.
test_daten = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")
test_daten = pd.DataFrame(test_daten)
test_daten = test_daten.drop(labels = ["Id"], axis=1)
test_daten = test_daten.values
ergebnisse = []
i=0
while i < len(test_daten):

    ergebnisse.append([i,klassifiziere(test_daten[i])])
    i = i+1
data_frame = pd.DataFrame(ergebnisse)
data_frame.columns = ["Id (String)", "Category (String)"]
data_frame.to_csv("submission.csv",index=False)
