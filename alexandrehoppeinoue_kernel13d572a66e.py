#Imports
import numpy as np
import pandas as pd
import urllib

import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plotter
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
import numpy

from sklearn.metrics import confusion_matrix
#Importando as base
trainData = pd.read_csv("../input/datafi/train_data.csv")
testData = pd.read_csv("../input/datafi/test_features.csv")

xTrain = trainData.iloc[:, 0:54] #excluo o ham e o id. Por isso -2
yTrain = trainData.iloc[:, -2]   #ham em -2
xTest = testData.iloc[:, 0:54] #no test não temos o ham
print(trainData.iloc[:,:])

#Ids da minha base de teste para colocar na saída ao final da predição
idsTest = testData.iloc[:, -1]  #id em -1
#print(idsTest)

trainData.describe()

#de coluna 0 até coluna 53
print(trainData.iloc[:,0:54])
trainData.iloc[:,0:54].describe()
#de coluna 54 até coluna 56
print(trainData.iloc[:,54:-2]) 
trainData.iloc[:,54:-2].describe()
#Gráficos de relação ham e feature
i = 0
while(i != 54):
    fig, axes = plotter.subplots()
    ttl = 'Feature %d' %(i)
    axes.set(xlabel='frequency of word', ylabel='Classified',title=ttl)
    t = xTrain.iloc[:, i]
    s = yTrain.iloc[:]
    plotter.scatter(t,s)
    plotter.show()
    i = i + 1 
    
    
#Correlação ham e feature
from math import sqrt
p =[]
j_p = []
i, j, num, den_1, den_2 = (0,)*5
y_barra = yTrain.iloc[:].values.mean()
while(j != 54):
    x_barra = trainData.iloc[:, j].values.mean()
    i,num, den_1, den_2 = (0,)*4
    while( i != yTrain.values.size):
        desvioX = trainData.iloc[i, j] - x_barra
        desvioY = (yTrain.iloc[i] - y_barra)*1000
        num = num + (desvioX)*(desvioY)
        den_1 = den_1 + desvioX**2
        den_2 = den_2 + desvioY**2
        i = i + 1
    p.append(num/(sqrt(den_1)*sqrt(den_2)))
    print("X_%d -> Coeficiente p = %f" %(j, p[j]))
    j_p.append([j, np.absolute(p[j])])
    j = j + 1 
    
#índice feature com valor de correlação (absolutos) impressos em ordem 
j_p = sorted(j_p, key=lambda j_p: j_p[1], reverse=True)
j_p = np.array(j_p)
t = j_p[:,0]
s = j_p[:,1]
fig, axes = plotter.subplots()
ttl = 'Correlação feature x classe '
axes.set(xlabel='Feature Xi', ylabel='Correlação com Classe',title=ttl)
plotter.scatter(t,s)
plotter.show()

j_p #primeira coluna id (em float...) segunda coluna coeficiente
#Primeira tentativa Gaussiana
gaussian = GaussianNB() #sem priors. Própria gaussian já faz
print(gaussian)
gaussian.fit(xTrain, yTrain)
scores = cross_val_score(gaussian, xTrain, yTrain, cv = 10) #cv = 10 => ten-fold validation
print(scores.mean())

#matriz de confusão
yPredTrain = cross_val_predict(gaussian, xTrain, yTrain, cv = 10)
print(confusion_matrix(yTrain, yPredTrain))
fpr, tpr, threshold = roc_curve(yTrain, yPredTrain)
print('Fp rate = ', fpr[1])
print('Tp rate = ', tpr[1])
#Rotina para achar onde binarizar
bin = 0.001
max = 0.9
maxBin = bin
while(bin < 1):
    bern = BernoulliNB(binarize = bin)
    bern.fit(xTrain, yTrain)
    scores = cross_val_score(bern, xTrain, yTrain, cv = 10) #cv = 10 => ten-fold validation
    if(scores.mean() > max):
        print(scores.mean())
        print (bin)
        max = scores.mean()
        maxBin = bin
    bin = bin + 0.001

bin = maxBin #0.1540
#Primeira tentativa Bernoulli
bern = BernoulliNB(binarize = bin)
bern.fit(xTrain, yTrain)
#acurácia
scores = cross_val_score(bern, xTrain, yTrain, cv = 10) #cv = 10 => ten-fold validation
print(scores.mean())

#matriz de confusão
yPredTrain = cross_val_predict(bern, xTrain, yTrain, cv = 10)
print(confusion_matrix(yTrain, yPredTrain))
fpr, tpr, threshold = roc_curve(yTrain, yPredTrain)
print('Fp rate:', fpr[1])
print('Tp rate:', tpr[1])
#Primeira tentativa Multinomial
multinomial = MultinomialNB()
print(multinomial)
multinomial.fit(xTrain, yTrain)
scores = cross_val_score(multinomial, xTrain, yTrain, cv = 10) #cv = 10 => ten-fold validation
print(scores.mean())

#matriz de confusão
yPredTrain = cross_val_predict(multinomial, xTrain, yTrain, cv = 10)
print(confusion_matrix(yTrain, yPredTrain))
fpr, tpr, threshold = roc_curve(yTrain, yPredTrain)
print('Fp rate:', fpr[1])
print('Tp rate:', tpr[1])
classificadorNaiveBayes = bern
from sklearn.neighbors import KNeighborsClassifier
#Achando hiperparâmetro: melhor valor de k
for n in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(xTrain, yTrain)
    scores = cross_val_score(knn, xTrain, yTrain, cv = 10)
    print(n, scores.mean())
#valor escolhido de n
n = 5
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(xTrain, yTrain)
scores = cross_val_score(knn, xTrain, yTrain, cv = 10)
print(scores.mean())

#matriz de confusão
yPredTrain = cross_val_predict(knn, xTrain, yTrain, cv = 10)
print(confusion_matrix(yTrain, yPredTrain))
fpr, tpr, threshold = roc_curve(yTrain, yPredTrain)
print('Fp rate:', fpr[1])
print('Tp rate:', tpr[1])
acc_vec = []
for i in range(1, 55):
    xTrainFinal = pd.DataFrame(trainData.iloc[:,j_p[0:i,0]]) # i melhores features
    classificadorNaiveBayes.fit(xTrainFinal, yTrain)
    #yPredTrain = cross_val_predict(classificadorNaiveBayes, xTrainFinal, yTrain, cv = 10)
    acc = cross_val_score(classificadorNaiveBayes, xTrainFinal, yTrain, cv = 10)
    acc_vec.append(acc.mean())
t = np.arange(1, 55, 1)
plotter.scatter(t, acc_vec)
plotter.show()
acc_vec

#30 melhores features
xTrainFinal = pd.DataFrame(trainData.iloc[:,j_p[0:30,0]]) 
#Rotina para achar onde binarizar
bin = 0.001
max = 0.9
while(bin < 1):
    bern = BernoulliNB(binarize = bin)
    bern.fit(xTrainFinal, yTrain)
    scores = cross_val_score(bern, xTrainFinal, yTrain, cv = 10) #cv = 10 => ten-fold validation
    if(scores.mean() > max):
        print(scores.mean())
        print (bin)
        max = scores.mean()
        maxBin = bin
    bin = bin + 0.001

bin = maxBin  #0.174
print(bin)
#30 primeiras melhores features NaiveBayes
classificadorNaiveBayes = BernoulliNB(binarize = 0.174)
classificadorNaiveBayes.fit(xTrainFinal, yTrain)
yPredTrain = cross_val_predict(classificadorNaiveBayes, xTrainFinal, yTrain, cv = 10)
fpr, tpr, threshold = roc_curve(yTrain, yPredTrain)
print(confusion_matrix(yTrain, yPredTrain))
scores = cross_val_score(classificadorNaiveBayes, xTrainFinal, yTrain, cv = 10)
print(scores.mean())


#Achando hiperparâmetro: melhor valor de vizinhos para as novas features
for n in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(xTrainFinal, yTrain)
    scores = cross_val_score(knn, xTrainFinal, yTrain, cv = 10)
    print(n, scores.mean())
#30 primeiras melhores features KNN
#valor de n = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrainFinal, yTrain)
yPredTrain2 = cross_val_predict(knn, xTrainFinal, yTrain, cv = 10)
fpr, tpr, threshold = roc_curve(yTrain, yPredTrain2)
print(confusion_matrix(yTrain, yPredTrain2))
scores = cross_val_score(knn, xTrainFinal, yTrain, cv = 10)
print(scores.mean())
#achando melhor binarize para um estimador com as features das capital letters
bin2 = 0.01
max2 = 0.6
while(bin2 < 10):
    bern = BernoulliNB(binarize = bin2)
    bern.fit(trainData.iloc[:,54:-2], yTrain)
    scores2 = cross_val_score(bern, trainData.iloc[:,54:-2], yTrain, cv = 10) #cv = 10 => ten-fold validation
    if(scores2.mean() > max2):
        print(scores2.mean())
        print (bin2)
        max2 = scores2.mean()
        maxBin2 = bin2
    bin2 = bin2 + 0.1

bin2 = maxBin2 


bern = BernoulliNB(binarize = 3.11)
bern.fit(trainData.iloc[:,54:-2], yTrain)
scores2 = cross_val_score(bern, trainData.iloc[:,54:-2], yTrain, cv = 10) #cv = 10 => ten-fold validation
print(scores2.mean())
#classificador misto
from sklearn.ensemble import VotingClassifier

classifMisto = VotingClassifier(estimators=[('naive', classificadorNaiveBayes), ('knn', knn)], voting='soft', weights = [4,8])
classifMisto.fit(xTrainFinal, yTrain)
print(cross_val_predict(classifMisto, xTrainFinal, yTrain, cv = 10))

#obs: o código gera um warning na biblioteca mais recente da sklearn pois tenta verificar se um vetor está vazio por if (vector)
#em vez de checar a length. Na próxima release da sklearn isso deverá estar normal.
classifMisto.fit(xTrainFinal, yTrain)
xTestFinal = pd.DataFrame(testData.iloc[:,j_p[0:30,0]]) #30 melhores features
yPred = classifMisto.predict(xTestFinal)
yPred1 = cross_val_predict(classifMisto, xTrainFinal, yTrain, cv = 10)

xTestFinal2 = pd.DataFrame(testData.iloc[:,54:-1])
yPred2 = bern.predict(trainData.iloc[:,54:-2])
yyPred = bern.predict(xTestFinal2)

classif = BernoulliNB()
classif.fit(pd.DataFrame(np.transpose([yPred1, yPred2])), yTrain)
yPred = classif.predict(np.transpose((pd.DataFrame([yPred, yyPred]))))
d = {'Id' : idsTest, 'Ham' : yPred}
my_df = pd.DataFrame(d) #import to dataframe
my_df.to_csv('prediction.csv',
             index=False, sep=',', line_terminator = '\n', header = ["Id", "Ham"])

#Resultado obtido
my_df