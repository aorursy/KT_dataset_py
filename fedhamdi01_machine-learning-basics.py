import sklearn 

import pandas as pd 

import matplotlib as plt

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings('ignore')
df=pd.read_excel("../input/german credit dataset.xls",header=0)
df.head()
print(df.shape)

df.dtypes


data=df.as_matrix()
X = data[:,0:20] #la matrice des variables explicatives

Y = data[:,20] #la variable a predire
from sklearn import model_selection #



X_app,X_test,y_app,y_test=model_selection.train_test_split(X,Y,test_size = 300,random_state=0)
print(X_app.shape, X_test.shape, y_app.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
modele_LR=LR.fit(X_app,y_app) #exécution de l'instance sur les données d'apprentissage

                       #c.à d . construction du modèle prédictif
print(modele_LR.coef_)
print(modele_LR.intercept_)
y_pred=modele_LR.predict(X_test) #Prediction sur l'echantillon test

#importation de metrics utilisé pour les mesures de performances

from sklearn import metrics
#matrice de confusion

#confrontation entre Y obs. sur l’éch . test et la prédiction

cm =metrics.confusion_matrix(y_test, y_pred)



cf = pd.DataFrame(cm, columns=['prédit ' + "0", 'predit'+"1"])

cf.index = ['vrai ' + "0",'vrai'+"1"]

cf
acc= metrics.accuracy_score(y_test,y_pred)

print(acc )
err= 1.0 - acc

print(err)
se=metrics.recall_score(y_test,y_pred)

print(se)
def specificity(y,y_hat):

    

    #matrice de confusion un objet numpy .ndarray

    mc = metrics.confusion_matrix(y,y_hat)

    #’’non solvable est sur l'indice 0

    import numpy

    res = mc [0,0]/numpy.sum(mc[0,:])

    #retour

    return res

#
specificite = metrics.make_scorer(specificity,greater_is_better=True)
sp = specificite(modele_LR,X_test,y_test)

print(sp)
probas=LR.predict_proba(X_test) #calcul des probas d'affectation sur ech . test

print(probas)

## Probas d'affectation aux classe "Non solvable" ~ 0 "Solvable" ~ 1
#score de presence



score = probas[:,1]

print(score)

pos = pd.get_dummies(y_test).as_matrix()
print(pos)
pos=pos[:,1]
print(pos)
import numpy as np

npos=np.sum(pos)
print(npos) # il y a 214 indiv solvable dans l'echantillon test
# indexe pour tri de slection 

index = np.argsort (score)

print(index)
index=index[::-1]
# tri des individus

sort_pos=pos[index]

sort_pos
#somme cumulé 

cpos=np.cumsum(sort_pos)

print(cpos)
# Rappel

rappel=cpos/npos

rappel
#nombre d observation dans l echantillon test

n=y_test.shape[0]

n
taille = np.arange(start =1,stop=301,step =1)
#passer en proportion

taille = taille / n
#titre et en têtes

plt.title ('Courbe de gain (lift cumulé)')

plt.xlabel ('Taille de cible')

plt.ylabel ('Rappel')

#limites en abscisse et ordonnée

plt.xlim (0,1)

plt.ylim (0,1)

#astuce pour tracer la diagonale

plt.scatter(taille,taille,marker='.', color= 'blue')

#insertion du couple (taille, rappel)

plt.scatter(taille,rappel,marker='.', color='red')

#affichage

plt.show
from sklearn.metrics import roc_auc_score, roc_curve, auc



fpr0, tpr0, thresholds0 = roc_curve(y_test, probas[:, 0], pos_label=modele_LR.classes_[0], drop_intermediate=False)

fpr0.shape
dftp = pd.DataFrame(dict(fpr=fpr0, tpr=tpr0, threshold=thresholds0)).copy()

dftp.head(n=2)
ax = dftp.plot(x="threshold", y=['fpr', 'tpr'], figsize=(10, 10))

ax.set_title("Evolution de FPR, TPR\nen fonction du seuil au delà duquel\n" +

             "la réponse du classifieur est validée");
fig, ax = plt.subplots(1, 1, figsize=(10,9))

ax.plot([0, 1], [0, 1], 'k--')

aucf = roc_auc_score(y_test == modele_LR.classes_[0], probas[:, 0]) # première façon

#aucf = auc(fpr0, tpr0)  # seconde façon

ax.plot(fpr0, tpr0, label=str(modele_LR.classes_[0] ) + '  ||| auc=%1.5f' % aucf)

ax.set_title('Courbe ROC - classifieur Solvabilité des client de Deutsh-Bank')

ax.text(0.5, 0.1, "plus mauvais que\nle hasard dans\ncette zone")

ax.legend();
aucf