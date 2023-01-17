# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn import model_selection

# Read some arff data

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


def true_accuracy(classes):
    right = 0
    for i,c in enumerate(classes):
        if i < 200:
            if c == -1: right += 1
        else:
            if c == 1: right += 1
    return right/len(classes)

trainingsset = read_data("../input/kiwhs-comp-1-complete/train.arff")

test_data = pd.read_csv('../input/kiwhs-comp-1-complete/test.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
np_test = test_data[['X','Y']].as_matrix()

data = [(x,y,c) for x,y,c in trainingsset]
data = np.array(data)
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)


#Finde den besten Wert für den Parameter tol für das Modell
def finde_besten_Wert(tr_x, te_x, tr_y, te_y):
    bester_parameterwert = 0.0
    bester_prozentwert = 0.0
    akt_prozentwert = 0.0
    akt_parameterwert = 0.1
    while(akt_parameterwert < 10):
        lr = linear_model.LogisticRegression(tol = akt_parameterwert)
        lr.fit(train_x, train_y)
        predict = lr.predict(test_x)
        i = 0 
        richtig = 0
        falsch = 0
        while(i < len(predict)):
            if(predict[i] == test_y[i]):
                richtig += 1
            else:
                falsch += 1
            i += 1
        akt_prozentwert = richtig/len(predict) * 100
        if (akt_prozentwert > bester_prozentwert):
            bester_parameterwert = akt_parameterwert
            bester_prozentwert = akt_prozentwert
        akt_parameterwert += 0.1
    return bester_prozentwert, bester_parameterwert


prozentwert,bester_parameterwert = finde_besten_Wert(train_x, test_x, train_y, test_y)
lr = linear_model.LogisticRegression(tol = bester_parameterwert)
lr.fit(train_x,train_y)

print("Das Ergebnis auf den alten Trainingsdaten mit Optimierung: ", prozentwert, "%")
print("Das Ergebnis auf den alten Testdaten mit Optimierung:",true_accuracy(lr.predict(np_test)) * 100,"%")


# Neue Daten einlesen
newData = read_data("../input/skewed/train-skewed.arff")

test_data = pd.read_csv('../input/skewed/test-skewed-header.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
np_test = test_data[['X','Y']].as_matrix()


data = [(x,y,c) for x,y,c in newData]
data = np.array(data)
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)

prozentwert, bester_parameterwert = finde_besten_Wert(train_x, test_x, train_y, test_y)

lr = linear_model.LogisticRegression(tol = bester_parameterwert)
lr.fit(train_x,train_y)

print("Das Ergebnis auf den neuen Trainingsdaten mit Optimierung: ", prozentwert, "%")
print("Das Ergebnis auf den neuen Testdaten mit Optimierung:",true_accuracy(lr.predict(np_test)) * 100,"%")
