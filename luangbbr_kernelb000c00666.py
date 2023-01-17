import sklearn
import sklearn.model_selection
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
import sklearn.naive_bayes as skNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#leitura dataset de treino
train_raw = pd.read_csv("train_data.csv", sep = ",")
display(train_raw.shape,
train_raw.head())
#leitura do dataset de test 
test_raw = pd.read_csv("test_features.csv", sep = ",",header=0)
display(test_raw.shape,
test_raw.head())

#tratamento treino
pd.options.mode.chained_assignment = None
target = train_raw["ham"]
train = train_raw.drop(["Id","ham"],axis = 1).copy()
#Drop non-freq features
lista=list(train_raw.corr()["ham"].abs().nlargest(34).keys())
trainN = train_raw.loc[:,lista].drop(["ham"],axis=1)
#Binarização
trainB = train_raw.loc[:,lista].drop(["ham"],axis=1)
for i in trainB:
    for j in range(len(trainB[i])):
        if trainB[i][j]>0:
            trainB[i][j]=1

#tratamento teste
test = test_raw.loc[:,lista].drop(["ham"],axis=1)
#Criando Naive Bayes

    #Multinomial Naive-Bayes
MNB = skNB.MultinomialNB()
    #Gaussian Naive-Bayes
GNB = skNB.GaussianNB()
    #Bernoulli Naive-Bayes
BNB = skNB.BernoulliNB()
def mostra(lista,nome):
    print(nome+": Média = "+str(round(np.mean(lista),4))+"  Desvio-Padrão = "+str(round(np.std(lista),4)))
fbeta3 = make_scorer(fbeta_score, beta=3)
#Comparacao de modelos (raw)

scoresM = sklearn.model_selection.cross_val_score(MNB, train, target, cv=10,scoring = fbeta3)
scoresG = sklearn.model_selection.cross_val_score(GNB, train, target, cv=10,scoring = fbeta3)
scoresB = sklearn.model_selection.cross_val_score(BNB, train, target, cv=10,scoring = fbeta3)
mostra(scoresM,"Multinomial")
mostra(scoresG,"Gaussian")
mostra(scoresB,"Bernoulli")
#Comparacao de modelos (drop non-freq)

scoresM = sklearn.model_selection.cross_val_score(MNB, trainN, target, cv=10,scoring = fbeta3)
scoresG = sklearn.model_selection.cross_val_score(GNB, trainN, target, cv=10,scoring = fbeta3)
scoresB = sklearn.model_selection.cross_val_score(BNB, trainN, target, cv=10,scoring = fbeta3)
mostra(scoresM,"Multinomial")
mostra(scoresG,"Gaussian")
mostra(scoresB,"Bernoulli")
#Comparacao de modelos (Binarizado)

scoresM = sklearn.model_selection.cross_val_score(MNB, trainB, target, cv=10,scoring = fbeta3)
scoresG = sklearn.model_selection.cross_val_score(GNB, trainB, target, cv=10,scoring = fbeta3)
scoresB = sklearn.model_selection.cross_val_score(BNB, trainB, target, cv=10,scoring = fbeta3)
mostra(scoresM,"Multinomial")
mostra(scoresG,"Gaussian")
mostra(scoresB,"Bernoulli")
#treinamento
nb=BNB.fit(trainN, target)
# KNN trial
KNN = sklearn.neighbors.KNeighborsClassifier(15)
cv=sklearn.model_selection.cross_val_score(KNN,trainB,target,cv=10,scoring = fbeta3)
mostra(cv,"KNN")
knnfit=KNN.fit(trainB,target)
#predict
R=knnfit.predict(test)
arq=open("result.csv","w")
arq.write("Id,ham\n")
for i,j in zip(list(R),list(test_raw["Id"].values)):
    arq.write(str(j)+","+str(i)+"\n")
arq.close()
    
    
"""scores_knn=[]
for i in range(2,56):
    lista=list(train_raw.corr()["ham"].abs().nlargest(i).keys()) #seleciona as i features de maior correlacao com ham,note que ham esta inclusa por corr de ham com ham =1
    trainN = train_raw.loc[:,lista].drop(["ham"],axis=1)
    #Binarização
    trainB = train_raw.loc[:,lista].drop(["ham"],axis=1)
    for i in trainB:
        for j in range(len(trainB[i])):
            if trainB[i][j]>0:
                trainB[i][j]=1
    KNN = sklearn.neighbors.KNeighborsClassifier(21)
    cv=sklearn.model_selection.cross_val_score(KNN,trainB,target,cv=10,scoring = fbeta3)
    scores_knn.append(cv.mean())

display(plt.scatter(list(range(2,56)),scores_knn),np.array(scores_knn).argmax()+5)
"""
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5)
classifier = KNN
X=trainB
y=target

thresholds = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    fpr, tpr, thr = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    thresholds.append(interp(mean_fpr, fpr, thr)) 
    thresholds[-1][0] = 1.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_thresholds = np.mean(thresholds, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC curve (AUC = %0.2f)' % (mean_auc),
         lw=2, alpha=.8)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Example ROC Curve')
plt.legend(loc="lower right")
plt.show()
