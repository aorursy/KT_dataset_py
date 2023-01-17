import numpy as np
import pandas as pd

import os
#print(os.listdir("../input/pmr-3508-tarefa-2/"))
spam = pd.read_csv("../input/spam-tarefa2/train_data.csv",header=0, index_col=-1)
print(spam.shape)
spam.head()
import matplotlib.pyplot as plt
%matplotlib inline
plt.grid()
plt.errorbar(range(54),y=spam.loc[spam['ham']==True].mean(axis=0).iloc[:-4], yerr=spam.loc[spam['ham']==True].std(axis=0).iloc[:-4], fmt='-o', label='ham')
plt.errorbar(range(54),y=spam.loc[spam['ham']==False].mean(axis=0).iloc[:-4], yerr=spam.loc[spam['ham']==False].std(axis=0).iloc[:-4], fmt='-o', label='spam')
plt.legend()
plt.grid()
plt.errorbar(range(54,57),y=spam.loc[spam['ham']==True].mean(axis=0).iloc[-4:-1], yerr=spam.loc[spam['ham']==True].std(axis=0).iloc[-4:-1], fmt='-o', label = 'ham')
plt.errorbar(range(54,57),y=spam.loc[spam['ham']==False].mean(axis=0).iloc[-4:-1], yerr=spam.loc[spam['ham']==False].std(axis=0).iloc[-4:-1], fmt='-o', label='spam')
plt.legend()
indices = [1,3,5,6,7,8,10,15,16,17,18,19]+list(range(21,27))+list(range(28,46))+[47,48]+list(range(51,54)) #Indices dos features
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
X = spam.iloc[:,indices]
y = spam['ham']
clf = BernoulliNB()
scores = cross_val_score(clf, X, y, cv=10)
print("Bernoulli Naive Bayes:")
print("Acurácia média: "+str(np.mean(scores)))
print("Desvio padrão: "+str(np.std(scores)))
from sklearn.naive_bayes import MultinomialNB
X = spam.iloc[:,indices]
y = spam['ham']
clf = MultinomialNB()
scores = cross_val_score(clf, X, y, cv=10)
print("Multinomial Naive Bayes:")
print("Acurácia média: "+str(np.mean(scores)))
print("Desvio padrão: "+str(np.std(scores)))
from sklearn.neighbors import KNeighborsClassifier
X = spam.iloc[:,indices]
y = spam['ham']
mean_scores = []
std_scores = []
for i in range(1,11):
    clf = KNeighborsClassifier(n_neighbors=i,p=1, weights='distance')
    scores = cross_val_score(clf, X, y, cv=10)
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))
print("KNN (validação cruzada realizada para k variando de 1 a 10):")
print("k para máxima acurácia média: "+str(np.argmax(mean_scores)+1))
print("Máxima acurácia média: "+str(np.amax(mean_scores)))
print("Desvio padrão para máxima acurácia média: "+str(std_scores[np.argmax(mean_scores)]))
plt.errorbar(range(1,11), mean_scores, yerr=1.96*np.array(std_scores), fmt='-o')
plt.title("Acurácia média da validação cruzada para valores de k")
plt.ylabel("Acurácia")
plt.xlabel("k")
test_feats = pd.read_csv("../input/spam-tarefa2/test_features.csv",header=0)
n_neighbors = 7
X = spam.iloc[:,indices]
y = spam['ham']
clf = KNeighborsClassifier(n_neighbors=n_neighbors,p=1, weights='distance')
#clf = BernoulliNB()
clf.fit(X,y) # Treinamento do classificador com base de treino
predicted_labels = clf.predict(test_feats.iloc[:,indices])
test_feats.head()
#test_feats = pd.read_csv("../input/spam-tarefa2/test_s.csv",header=0)
prediction = pd.read_csv("../input/spam-tarefa2/sample_submission_1.csv",header=0)
prediction['ham'] = predicted_labels
#prediction.index.rename('Id',inplace=True)
prediction.head()
prediction.to_csv("submission.csv", index=False)
from sklearn.model_selection import train_test_split
X = spam.iloc[:,indices]
y = spam['ham']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state=1)
clf = KNeighborsClassifier(n_neighbors=n_neighbors,p=1, weights='distance')
#clf = BernoulliNB()
clf.fit(X_train,y_train)
y_pred = clf.predict_proba(X_test)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1], pos_label=True) #TODO validação cruzada!!!
plt.plot(fpr,tpr)
plt.title("Curva ROC")
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
from sklearn.metrics import fbeta_score
f_beta = []
for i in range(11):
    f_beta.append(fbeta_score(y_test, y_pred[:,1]>=i*0.1, beta=3))
plt.plot([0.1*i for i in range(11)],f_beta)
plt.title("F_beta para beta = 3")
plt.xlabel('Threshold de probabilidade')
plt.ylabel('F_beta')
print("F_beta max : "+str(np.amax(f_beta)))
print("Threshold de probabilidade para F_beta max : "+str(0.1*np.argmax(f_beta)))
print("FPR esperado para threshold de F_beta max : "+str(fpr[np.argmin(np.abs(thresholds-0.1*np.argmax(f_beta)))]))