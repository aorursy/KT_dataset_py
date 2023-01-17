from __future__ import print_function, division

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings

import matplotlib.patches as mpatches

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.utils import check_random_state

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

rs = check_random_state(4421)
data = pd.read_csv("../input/league-of-legends/games.csv")
partidasT1 = data[['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','winner']]

partidasT2 = data[['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills','winner']]

partidasT1['winner'] = partidasT1['winner'].replace({2: 0}, regex=True)

partidasT2['winner'] = partidasT2['winner'].replace({1: 0}, regex=True)

partidasT2['winner'] = partidasT2['winner'].replace({2: 1}, regex=True)



partidasT2['firstBlood'] = partidasT2['firstBlood'].replace({2: 3}, regex=True)

partidasT2['firstBlood'] = partidasT2['firstBlood'].replace({1: 2}, regex=True)

partidasT2['firstBlood'] = partidasT2['firstBlood'].replace({3: 1}, regex=True)



partidasT2['firstTower'] = partidasT2['firstTower'].replace({2: 3}, regex=True)

partidasT2['firstTower'] = partidasT2['firstTower'].replace({1: 2}, regex=True)

partidasT2['firstTower'] = partidasT2['firstTower'].replace({3: 1}, regex=True)



partidasT2['firstInhibitor'] = partidasT2['firstInhibitor'].replace({2: 3}, regex=True)

partidasT2['firstInhibitor'] = partidasT2['firstInhibitor'].replace({1: 2}, regex=True)

partidasT2['firstInhibitor'] = partidasT2['firstInhibitor'].replace({3: 1}, regex=True)



partidasT2['firstBaron'] = partidasT2['firstBaron'].replace({2: 3}, regex=True)

partidasT2['firstBaron'] = partidasT2['firstBaron'].replace({1: 2}, regex=True)

partidasT2['firstBaron'] = partidasT2['firstBaron'].replace({3: 1}, regex=True)



partidasT2['firstDragon'] = partidasT2['firstDragon'].replace({2: 3}, regex=True)

partidasT2['firstDragon'] = partidasT2['firstDragon'].replace({1: 2}, regex=True)

partidasT2['firstDragon'] = partidasT2['firstDragon'].replace({3: 1}, regex=True)



partidasT2['firstRiftHerald'] = partidasT2['firstRiftHerald'].replace({2: 3}, regex=True)

partidasT2['firstRiftHerald'] = partidasT2['firstRiftHerald'].replace({1: 2}, regex=True)

partidasT2['firstRiftHerald'] = partidasT2['firstRiftHerald'].replace({3: 1}, regex=True)

partidasT1.columns = ['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','towerKills' ,'inhibitorKills','baronKills','dragonKills','riftHeraldKills','winner']

partidasT2.columns = ['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','towerKills' ,'inhibitorKills','baronKills','dragonKills','riftHeraldKills','winner']

partidas= pd.concat([partidasT1,partidasT2],axis=0)

partidas.index = pd.RangeIndex(len(partidas.index))

partidas['gameDuration'] = (partidas.gameDuration/60).astype(int)
plt.figure(figsize=(15, 5))

plt.subplot(121)

plt.hist(partidas["gameDuration"],bins =25,color = "skyblue",edgecolor='blue',linewidth=1)

plt.xlabel('Minutos')

plt.ylabel('Número de partidas')

plt.title('Duración de las partidas')

plt.subplot(122)

plt.ylabel('Minutos')

plt.title('Gráfico de caja y bigotes')

plt.boxplot(partidas["gameDuration"])

plt.show()

partidas["gameDuration"].describe()

partidas['gameDuration'] = np.where(partidas['gameDuration']<25,1,np.where(((partidas['gameDuration']>24)&(partidas['gameDuration']<36)),2,3))
killsWinner = partidas.drop(partidas[partidas.loc[:, 'winner'] == 0].index)

killsWinner.index = pd.RangeIndex(len(killsWinner.index))

LoserKills = partidas.drop(partidas[partidas.loc[:, 'winner'] == 1].index)

LoserKills.index = pd.RangeIndex(len(LoserKills.index))
firstBloodfi = killsWinner.firstBlood.value_counts()/killsWinner.firstBlood.count()

firstBloodfi =firstBloodfi.to_frame()

firstBloodfi['index'] = firstBloodfi.index



firstTowerfi = killsWinner.firstTower.value_counts()/killsWinner.firstTower.count()

firstTowerfi =firstTowerfi.to_frame()

firstTowerfi['index'] = firstTowerfi.index



firstInhibitorfi = killsWinner.firstInhibitor.value_counts()/killsWinner.firstInhibitor.count()

firstInhibitorfi =firstInhibitorfi.to_frame()

firstInhibitorfi['index'] = firstInhibitorfi.index



firstBaronfi = killsWinner.firstBaron.value_counts()/killsWinner.firstBaron.count()

firstBaronfi =firstBaronfi.to_frame()

firstBaronfi['index'] = firstBaronfi.index





firstDragonfi = killsWinner.firstDragon.value_counts()/killsWinner.firstDragon.count()

firstDragonfi =firstDragonfi.to_frame()

firstDragonfi['index'] = firstDragonfi.index



firstRiftHeraldfi = killsWinner.firstRiftHerald.value_counts()/killsWinner.firstRiftHerald.count()

firstRiftHeraldfi =firstRiftHeraldfi.to_frame()

firstRiftHeraldfi['index'] = firstRiftHeraldfi.index





inhibitorKillsfi = killsWinner.inhibitorKills.value_counts()/killsWinner.inhibitorKills.count()

inhibitorKillsfi =inhibitorKillsfi.to_frame()

inhibitorKillsfi['index'] = inhibitorKillsfi.index



towerKillsfi = killsWinner.towerKills.value_counts()/killsWinner.towerKills.count()

towerKillsfi =towerKillsfi.to_frame()

towerKillsfi['index'] = towerKillsfi.index



baronKillsfi = killsWinner.baronKills.value_counts()/killsWinner.baronKills.count()

baronKillsfi =baronKillsfi.to_frame()

baronKillsfi['index'] = baronKillsfi.index





dragonKillsfi =killsWinner.dragonKills.value_counts()/killsWinner.dragonKills.count()

dragonKillsfi =dragonKillsfi.to_frame()

dragonKillsfi['index'] = dragonKillsfi.index





riftHeraldKillsfi =killsWinner.riftHeraldKills.value_counts()/killsWinner.riftHeraldKills.count()

riftHeraldKillsfi =riftHeraldKillsfi.to_frame()

riftHeraldKillsfi['index'] = riftHeraldKillsfi.index

inhibitorKillsfi2 = LoserKills.inhibitorKills.value_counts()/LoserKills.inhibitorKills.count()

inhibitorKillsfi2 =inhibitorKillsfi2.to_frame()

inhibitorKillsfi2['index'] = inhibitorKillsfi2.index



towerKillsfi2 = LoserKills.towerKills.value_counts()/LoserKills.towerKills.count()

towerKillsfi2 =towerKillsfi2.to_frame()

towerKillsfi2['index'] = towerKillsfi2.index



baronKillsfi2 = LoserKills.baronKills.value_counts()/LoserKills.baronKills.count()

baronKillsfi2 =baronKillsfi2.to_frame()

baronKillsfi2['index'] = baronKillsfi2.index





dragonKillsfi2 =LoserKills.dragonKills.value_counts()/LoserKills.dragonKills.count()

dragonKillsfi2 =dragonKillsfi2.to_frame()

dragonKillsfi2['index'] = dragonKillsfi2.index





riftHeraldKillsfi2 =LoserKills.riftHeraldKills.value_counts()/LoserKills.riftHeraldKills.count()

riftHeraldKillsfi2 =riftHeraldKillsfi2.to_frame()

riftHeraldKillsfi2['index'] = riftHeraldKillsfi2.index
plt.figure(figsize=(15, 5))

plt.subplot(131).set_title('Primera Muerte')

plt.bar(firstBloodfi.index-0.125,firstBloodfi.firstBlood,width = 0.25,color='blue')

plt.subplot(132).set_title('Primera Torre')

plt.bar(firstTowerfi.index-0.125,firstTowerfi.firstTower,width = 0.25,color='blue')

plt.subplot(133).set_title('Primer Inhibidor')

plt.bar(firstInhibitorfi.index-0.125,firstInhibitorfi.firstInhibitor,width = 0.25,color='blue')

plt.show()



plt.figure(figsize=(15, 5))

plt.subplot(131).set_title('Primer Barón')

plt.bar(firstBaronfi.index-0.125,firstBaronfi.firstBaron,width = 0.25,color='blue')

plt.subplot(132).set_title('Primer Dragón')

plt.bar(firstDragonfi.index-0.125,firstDragonfi.firstDragon,width = 0.25,color='blue')

plt.subplot(133).set_title('Primer Heraldo')

plt.bar(firstRiftHeraldfi.index-0.125,firstRiftHeraldfi.firstRiftHerald,width = 0.25,color='blue')

plt.show()



plt.figure(figsize=(15, 5))

plt.subplot(131).set_title('Inhibidores Destruidos')

blue_legend = mpatches.Patch(color='blue', label='Ganadores')

yellow_legend = mpatches.Patch(color='yellow', label='Perdedores')

plt.legend(handles=[blue_legend,yellow_legend])

plt.bar(inhibitorKillsfi.index-0.125,inhibitorKillsfi.inhibitorKills,width = 0.25,color='blue')

plt.bar(inhibitorKillsfi2.index+0.125,inhibitorKillsfi2.inhibitorKills,width = 0.25,color='yellow')

plt.subplot(132).set_title('Torres Destruidas')

plt.legend(handles=[blue_legend,yellow_legend])

plt.bar(towerKillsfi.index-0.125,towerKillsfi.towerKills,width = 0.25,color='blue')

plt.bar(towerKillsfi2.index+0.125,towerKillsfi2.towerKills,width = 0.25,color='yellow')

plt.subplot(133).set_title('Barones Muertos')

plt.legend(handles=[blue_legend,yellow_legend])

plt.bar(baronKillsfi.index-0.125,baronKillsfi.baronKills,width = 0.25,color='blue')

plt.bar(baronKillsfi2.index+0.125,baronKillsfi2.baronKills,width = 0.25,color='yellow')

plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(131).set_title('Heraldos Muertos')

plt.legend(handles=[blue_legend,yellow_legend])

plt.bar(riftHeraldKillsfi.index-0.125,riftHeraldKillsfi.riftHeraldKills,width = 0.25,color='blue')

plt.bar(riftHeraldKillsfi2.index+0.125,riftHeraldKillsfi2.riftHeraldKills,width = 0.25,color='yellow')

plt.subplot(132).set_title('Dragones Muertos')

plt.legend(handles=[blue_legend,yellow_legend])

plt.bar(dragonKillsfi.index-0.125,dragonKillsfi.dragonKills,width = 0.25,color='blue')

plt.bar(dragonKillsfi2.index+0.125,dragonKillsfi2.dragonKills,width = 0.25,color='yellow')

plt.show()
plt.figure(figsize=(16, 16))

plt.title('Matriz de correlación')

sns.heatmap(partidas.corr(), square=True, annot=True,cmap="YlGnBu")
X = partidas.drop(['winner'], axis=1)

Y = partidas['winner']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=rs)
fpr = dict()

tpr = dict()

roc_auc = dict()
rango = range(1,30)

score = 0

precisiones = []

for vecinos in rango:

    knn = KNeighborsClassifier(n_neighbors = vecinos)

    knn.fit(Xtrain,Ytrain)

    if (score < knn.score(Xtest,Ytest)):

        Y_pred = knn.predict(Xtest)

        score = knn.score(Xtest,Ytest)

        pos = vecinos

    precisiones.insert(vecinos,knn.score(Xtest,Ytest))

print("precisión de predicción: {0: .3f}".format(score))

print("n_neighbors: "+format(vecinos))
plt.title('Precisión frente n_neighbors')

plt.ylabel('precisión')

plt.xlabel('n_neighbors')

plt.plot(precisiones)
confusion = confusion_matrix(Ytest, Y_pred)

plt.clf()

plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues_r)

classNames = ['Negativo','Positivo']

plt.title('Matriz de confusión k-Nearest Neighbor')

plt.ylabel('Valor verdadero')

plt.xlabel('Valor Predicho')



tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['VN','FP'], ['FN', 'VP']]

s = [['VN','FP'], ['FN', 'VP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i,str(s[i][j])+" = "+str(confusion[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()
fpr[1], tpr[1], _ = roc_curve(Ytest, Y_pred,pos_label=2)

roc_auc[1] = auc(fpr[1], tpr[1])
nb = GaussianNB()

nb.fit(Xtrain,Ytrain)

prediccion = nb.predict(Xtest)

accuracy_test = accuracy_score(Ytest,prediccion.round())

print( "Accuracy (test)  =", accuracy_test )
Y_pred = nb.predict(Xtest)

confusion = confusion_matrix(Ytest, Y_pred)

plt.clf()

plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues_r)

classNames = ['Negativo','Positivo']

plt.title('Matriz de confusión Naive Bayes')

plt.ylabel('Valor verdadero')

plt.xlabel('Valor Predicho')



tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['VN','FP'], ['FN', 'VP']]

s = [['VN','FP'], ['FN', 'VP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i,str(s[i][j])+" = "+str(confusion[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()
fpr[2], tpr[2], _ = roc_curve(Ytest, Y_pred,pos_label=2)

roc_auc[2] = auc(fpr[2], tpr[2])
lr = LogisticRegression(C=0.1,penalty='l2')

lr.fit(Xtrain,Ytrain)
Y_pred = lr.predict(Xtest)
accuracy_test = accuracy_score(Ytest,Y_pred.round())

print( "Accuracy (test)  =", accuracy_test )
confusion = confusion_matrix(Ytest, Y_pred)

plt.clf()

plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues_r)

classNames = ['Negativo','Positivo']

plt.title('Matriz de confusión Regresión Logística')

plt.ylabel('Valor verdadero')

plt.xlabel('Valor Predicho')



tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['VN','FP'], ['FN', 'VP']]

s = [['VN','FP'], ['FN', 'VP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i,str(s[i][j])+" = "+str(confusion[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()
fpr[3], tpr[3], _ = roc_curve(Ytest, Y_pred,pos_label=2)

roc_auc[3] = auc(fpr[3], tpr[3])
rango1 = range(2,40)

rango2 = range(1,5)

score = 0

precisiones = []



for estimadores in rango1:

    for estimadores2 in rango2:

        dt = DecisionTreeRegressor(random_state = rs,min_samples_split = estimadores,min_samples_leaf=estimadores2)

        dt.fit(Xtrain,Ytrain)

        prediccion = dt.predict(Xtest)

        accuracy_test = accuracy_score(Ytest,prediccion.round())

        if (score < accuracy_test):

            Y_pred = prediccion

            score = accuracy_test

            pos1= estimadores

            pos2= estimadores2

        precisiones.insert(estimadores,accuracy_test)

print("precisión de predicción: {0: .3f}".format(score))

print("min_samples_split: " +format(pos1))

print("min_samples_split: " +format(pos2))
plt.title('Evolución precisión con los distintos parámetros de entrada')

plt.ylabel('precisión')

plt.xlabel('Nº de ejecución')

plt.plot(precisiones)
Y_pred = Y_pred.round()

confusion = confusion_matrix(Ytest, Y_pred)

plt.clf()

plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues_r)

classNames = ['Negativo','Positivo']

plt.title('Matriz de confusión Árbol de decisión')

plt.ylabel('Valor verdadero')

plt.xlabel('Valor Predicho')



tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['VN','FP'], ['FN', 'VP']]

s = [['VN','FP'], ['FN', 'VP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i,str(s[i][j])+" = "+str(confusion[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()
fpr[4], tpr[4], _ = roc_curve(Ytest, Y_pred,pos_label=2)

roc_auc[4] = auc(fpr[4], tpr[4])
rango = range(1,500)

score = 0

precisiones = []

for estimadores in rango:

    clf = RandomForestClassifier(n_estimators=estimadores, random_state=rs)

    clf.fit(Xtrain,Ytrain)

    prediccion = clf.predict(Xtest)

    accuracy_test = accuracy_score(Ytest,prediccion.round())

    if (score < accuracy_test):

        Y_pred = prediccion

        score = accuracy_test

        pos = estimadores

    precisiones.insert(estimadores,accuracy_test)

print("precisión de predicción: {0: .3f}".format(score))

print("n_estimators: " +format(pos))
plt.title('Precisión frente a n_estimators')

plt.ylabel('precisión')

plt.xlabel('n_estimators')

plt.plot(precisiones)
confusion = confusion_matrix(Ytest, Y_pred)

plt.clf()

plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues_r)

classNames = ['Negativo','Positivo']

plt.title('Matriz de confusión Random Forest')

plt.ylabel('Valor verdadero')

plt.xlabel('Valor Predicho')



tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['VN','FP'], ['FN', 'VP']]

s = [['VN','FP'], ['FN', 'VP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i,str(s[i][j])+" = "+str(confusion[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()
fpr[5], tpr[5], _ = roc_curve(Ytest, Y_pred,pos_label=2)

roc_auc[5] = auc(fpr[5], tpr[5])
rango = range(1,200)

score = 0

precisiones = []

for estimadores in rango:

    gbr = GradientBoostingRegressor(n_estimators = estimadores,max_depth=7,min_samples_split=3,min_samples_leaf=2,max_features=2,random_state=rs)

    gbr.fit(Xtrain,Ytrain)

    prediccion = gbr.predict(Xtest)

    accuracy_test = accuracy_score(Ytest,prediccion.round())

    if (score < accuracy_test):

        Y_pred = prediccion

        score = accuracy_test

        pos = estimadores

    precisiones.insert(estimadores,accuracy_test)

print("precisión de predicción: {0: .3f}".format(score))

print("n_estimators: " +format(pos))
plt.title('Precisión frente a n_estimators')

plt.ylabel('precisión')

plt.xlabel('n_estimators')

plt.plot(precisiones)
confusion = confusion_matrix(Ytest, Y_pred.round())

plt.clf()

plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues_r)

classNames = ['Negativo','Positivo']

plt.title('Matriz de confusión Gradient Boosting')

plt.ylabel('Valor verdadero')

plt.xlabel('Valor Predicho')



tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['VN','FP'], ['FN', 'VP']]

s = [['VN','FP'], ['FN', 'VP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i,str(s[i][j])+" = "+str(confusion[i][j]), horizontalalignment='center', verticalalignment='center')

plt.show()
fpr[6], tpr[6], _ = roc_curve(Ytest, Y_pred,pos_label=2)

roc_auc[6] = auc(fpr[6], tpr[6])
plt.figure()

lw = 2

plt.plot(fpr[1], tpr[1], color='darkorange',lw=lw, label='Curva ROC k-Nearest Neighbor (area = %0.2f)' % roc_auc[1])

plt.plot(fpr[2], tpr[2], color='blue', lw=lw, label='Curva ROC  Naive Bayes (area = %0.2f)' % roc_auc[2])

plt.plot(fpr[3], tpr[3], color='green', lw=lw, label='Curva ROC  Regresion logistica (area = %0.2f)' % roc_auc[3])

plt.plot(fpr[4], tpr[4], color='yellow', lw=lw, label='Curva ROC  Árbol de decisión (area = %0.2f)' % roc_auc[4])

plt.plot(fpr[5], tpr[5], color='purple', lw=lw, label='Curva ROC  Random forest (area = %0.2f)' % roc_auc[5])

plt.plot(fpr[6], tpr[6], color='red', lw=lw, label='Curva ROC curve Gradient Boosting (area = %0.2f)' % roc_auc[6])



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()