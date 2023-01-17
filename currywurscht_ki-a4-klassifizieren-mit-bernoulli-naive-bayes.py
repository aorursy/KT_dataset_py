import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import BernoulliNB
from sklearn import model_selection
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
#Trainingsdaten laden
org_data = read_data("../input/kiwhs-comp-1-complete/train.arff")
data = [(x,y,c) for x,y,c in org_data]
data = np.array(data)
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)
print(len(train_x),len(test_x))
#Testdaten laden
test_data = pd.read_csv('../input/kiwhs-comp-1-complete/test.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
np_test = test_data[['X','Y']].as_matrix()
#Labe
label = []
for i in range(0,200):
    label.append(-1)
for i in range(200,399) :
    label.append(1)    
#Auswertungsfunktion
#Freie Interpretation von sklearn.naive_bayes.BernoulliNB.score
def score(testsamples,truelabelx):
    accuracy =0
    correct = 0
    for i in range(0, len(truelabelx)):
        accuracy+=1
        if testsamples[i] == truelabelx[i]:
            correct+=1
    return correct / accuracy  
nb = BernoulliNB()
nb.fit(train_x,train_y)
pred = nb.predict(test_x)
#print(nb.score(pred,test_y))
print("Score = %f "%(score(pred,test_y)))
def optimize():
    #bester/maximaler score
    maximum = 0
    #zaehler fuer den alphawert
    i = 0.1
    #zaehler fuer den binarize wert
    bina = 0.01
    fitprior = [False,True]
    while i <= 2:
        while bina <= 10 :
            for fit in range(0, len(fitprior), 1):
                berno = BernoulliNB(alpha = i, binarize = bina,fit_prior = fit)
                berno.fit(train_x,train_y)
                pred = berno.predict(train_x)
                blubb = score(pred,train_y)   
                #print(blubb)
                #print(" Score", blubb,"binarize = ", bina," fit_prior = ", fitprior[fit]," class_prior = ",classprior[clas]) 
                if blubb > maximum:
                    maximum = blubb 
                    best = bina
                    alpha = i
                    bestprior = fitprior[fit]
                    #print("h",best,"alpha",i,"fit",fitprior[fit])
            bina = bina + 0.01
        i = i + 0.01 
    return alpha, best,bestprior                                    
print(optimize())                               
#Anwendung auf die Testdaten aus train.arff
a,b,f =  optimize()
nb2 = BernoulliNB(alpha = a,binarize = b, fit_prior = f)
nb2.fit(train_x,train_y)
pred = nb2.predict(test_x)
print(score(pred,test_y))
#Test auf den Testdaten
nb2.predict(np_test)
score(nb2.predict(np_test),label)
#neue daten
new_train_data = read_data("../input/skewed/train-skewed.arff")
data = [(x,y,c) for x,y,c in new_train_data]
data = np.array(data)
train_x, test_x, train_y, test_y = model_selection.train_test_split(data[:,0:2], data[:,2], random_state = 1000)
new_test_data = pd.read_csv('../input/skewed/test-skewed-header.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
new_np_test = new_test_data[['X','Y']].as_matrix()

#Neue Daten testen
a2,b2,f2 =  optimize()
nb3 = BernoulliNB(alpha = a2,binarize = b2, fit_prior = f2)
nb3.fit(train_x,train_y)
pred2 = nb3.predict(test_x)
print(score(pred2,test_y))
pred2 = nb3.predict(new_np_test)
#Spoiler (für das Messen der von Ihnen erzielten Accuracy): ID 0-199 sind Klasse -1, ID 200-399 sind ID 1
# also muessen fuer die skewed daten keine neuen Labels mehr erstellt werden.
print(score(pred2,label))