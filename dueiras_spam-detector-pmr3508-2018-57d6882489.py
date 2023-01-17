import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer

data_treino_original = pd.read_csv('../input/train-data/train_data.csv', engine='python')

data_teste = pd.read_csv('../input/testefeat/test_features.csv', engine='python')
data_treino = data_treino_original.drop('ham', axis=1)

data_treino = data_treino.drop('Id', axis=1)

labels_treino = data_treino_original['ham']

id_teste = data_teste['Id']
data_treino_original.head()
    
data_treino.shape
mnb = MultinomialNB()
mnb.fit(data_treino, labels_treino)
mnb_predictions_treino = mnb.predict(data_treino)
print (accuracy_score(labels_treino, mnb_predictions_treino))
print (classification_report(labels_treino, mnb_predictions_treino))
print (confusion_matrix(labels_treino, mnb_predictions_treino))
cross_val = cross_val_score(mnb, data_treino, labels_treino, cv=10, scoring='accuracy')
print (cross_val)
print (np.mean(cross_val))
fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(mnb, data_treino, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
gnb = GaussianNB()
gnb.fit (data_treino, labels_treino)
gnb_predictions_treino = gnb.predict(data_treino)
print (accuracy_score(labels_treino, gnb_predictions_treino))
print (classification_report(labels_treino, gnb_predictions_treino))
print (confusion_matrix(labels_treino, gnb_predictions_treino))
cross_val = cross_val_score(gnb, data_treino, labels_treino, cv=10, scoring='accuracy')
print (cross_val)
print (np.mean(cross_val))
fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(gnb, data_treino, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
binary_data_treino = data_treino

pd.options.mode.chained_assignment = None
for i in binary_data_treino:
    for j in range(len(binary_data_treino[i])):
        if binary_data_treino[i][j]>0:
            binary_data_treino[i][j]=1
binary_data_treino.head()
bnb = BernoulliNB()
bnb.fit(binary_data_treino, labels_treino)
bnb_predictions_treino = bnb.predict(data_treino)
print (accuracy_score(labels_treino, bnb_predictions_treino))
print (classification_report(labels_treino, bnb_predictions_treino))
print (confusion_matrix(labels_treino, bnb_predictions_treino))
cross_val = cross_val_score(bnb, data_treino, labels_treino, cv=15, scoring='accuracy')
print (cross_val)
print (np.mean(cross_val))
fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(bnb, data_treino, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
y = []
x = range(8, 30)
for i in x:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, binary_data_treino, labels_treino, cv=10)
    y.append(scores.mean())
plt.scatter(x, y)
y = []
x = range(2, 15)
for i in x:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, binary_data_treino, labels_treino, cv=10)
    y.append(scores.mean())
plt.scatter(x, y)
y = []
x = range(8, 30)
for i in x:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, data_treino, labels_treino, cv=10)
    y.append(scores.mean())
plt.scatter(x, y)
knn = KNeighborsClassifier(n_neighbors=13)
scores = cross_val_score(knn, data_treino, labels_treino, cv=10)
display(scores)
scores.mean()
fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(knn, data_treino, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
scores = cross_val_score(knn, binary_data_treino, labels_treino, cv=10)
display(scores)
scores.mean()
fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(knn, binary_data_treino, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, data_treino, labels_treino, cv=10)
display(scores)
scores.mean()
fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(knn, binary_data_treino, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
data_treino_original.corr()['ham']
scores_knn=[]
for i in range(2,56):
    lista=list(data_treino_original.corr()["ham"].abs().nlargest(i).keys())
    trainN = data_treino_original.loc[:,lista].drop(["ham"],axis=1)
    
    trainB = data_treino_original.loc[:,lista].drop(["ham"],axis=1)
    for i in trainB:
        for j in range(len(trainB[i])):
            if trainB[i][j]>0:
                trainB[i][j]=1
                
    cv=sklearn.model_selection.cross_val_score(knn,trainB,labels_treino,cv=10,scoring = fbeta3)
    scores_knn.append(cv.mean())

display(plt.scatter(list(range(2,56)),scores_knn),np.array(scores_knn).argmax()+5)
lista2=list(data_treino_original.corr()["ham"].abs().nlargest(34).keys())
lista2.remove('ham')
print(lista2)
data_treino_p = data_treino
colunas = list(data_treino.columns)
for i in colunas:
    if i not in lista2:
        data_treino_p = data_treino_p.drop(i, axis=1)
mnb = MultinomialNB()
mnb.fit(data_treino_p, labels_treino)

fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(mnb, data_treino_p, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
gnb = GaussianNB()
gnb.fit (data_treino_p, labels_treino)

fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(gnb, data_treino_p, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
binary_data_treino_p = data_treino_p

pd.options.mode.chained_assignment = None
for i in binary_data_treino_p:
    for j in range(len(binary_data_treino_p[i])):
        if binary_data_treino_p[i][j]>0:
            binary_data_treino_p[i][j]=1
binary_data_treino_p.head()
bnb = BernoulliNB()
bnb.fit(binary_data_treino_p, labels_treino)

fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(bnb, binary_data_treino_p, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, data_treino_p, labels_treino, cv=10)
display(scores)
scores.mean()

fbeta3 = make_scorer (fbeta_score, beta = 3)
scores = sklearn.model_selection.cross_val_score(knn, data_treino_p, labels_treino, cv = 10, scoring = fbeta3)
print(scores.mean())
data_teste_p = data_teste
colunas = list(data_teste.columns)
for i in colunas:
    if i not in lista2:
        data_teste_p = data_teste_p.drop(i, axis=1)
        
data_teste_p2 = data_teste
colunas = list(data_teste.columns)
for i in colunas:
    if i != 'Id':
        if i not in lista2:
            data_teste_p2 = data_teste_p2.drop(i, axis=1)
knn.fit(data_treino_p,labels_treino)
labels_teste = knn.predict(data_teste_p)
display(labels_teste)
arq = open ("spamsubmission.csv", "w")
arq.write("Id,ham\n")
for i, j in zip(data_teste_p2['Id'], labels_teste):
    arq.write(str(i)+ "," + str(j)+"\n")
arq.close()
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

X = data_treino
y = labels_treino

# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
classifier = BernoulliNB()

thresholds = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
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
cv = StratifiedKFold(n_splits=5)
classifier = KNeighborsClassifier(n_neighbors=3)

thresholds = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
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