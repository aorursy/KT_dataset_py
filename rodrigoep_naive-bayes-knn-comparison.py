import pandas as pd
import sklearn
import numpy as np
from matplotlib import pyplot as plt
import math
import os
os.listdir('../input/')
train = pd.read_csv("../input/spambase/train_data.csv")
test = pd.read_csv("../input/spambase/test_features.csv")
Id = test.pop("Id")
Id_train = train.pop("Id")
train.head()
test.shape
train.info()
def correlation_matrix(df):
    
    from matplotlib import cm as cm
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Text Feature Correlation')
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.rcParams['figure.figsize'] = [20,5]
    plt.show()

correlation_matrix(train)
lista = list(train)
for i in lista:
    if(abs(train[i].corr(train['ham'])) <= 0.20):
        train.pop(i)
        test.pop(i)
Target = train.pop('ham')
train.shape
train.head()
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
bnb = BernoulliNB()
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
f3_scorer = make_scorer(fbeta_score, beta=3)
scores = cross_val_score(bnb, train, Target, cv=10, scoring = f3_scorer)
print("Score F3 atingido", scores.mean())
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(bnb, train, Target, cv=10)
cm = confusion_matrix(Target, y_pred)
print(cm)
from sklearn.metrics import roc_curve, auc

proba = cross_val_predict(bnb, train, Target, cv=10, method = 'predict_proba') 
fpr, tpr, threshold = roc_curve(Target, proba[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
dist = 1000
for i in range(len(tpr)):
    dist1 = math.sqrt((tpr[i]-1)**2 + (fpr[i])**2)
    if dist1 < dist:
        dist = dist1
        indice = i
print("melhor indice:",indice)
print("false positive rate deste indice:", fpr[indice])
print("melhor limite de probabilidades:", threshold[indice])
bnb = BernoulliNB(class_prior = [1-threshold[indice], threshold[indice]])
scores = cross_val_score(bnb, train, Target, cv=10, scoring = f3_scorer)
print("Score F3 atingido", scores.mean())
y_pred = cross_val_predict(bnb, train, Target, cv=10)
cm = confusion_matrix(Target, y_pred)
print(cm)
bnb.fit(train,Target)
predict = bnb.predict(test)
output =pd.DataFrame(Id)
output["ham"] = predict
output.to_csv("NaiveBayes", index = False)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
train = sc.fit_transform(train)  
test = sc.transform(test)  
from sklearn.decomposition import PCA
pca = PCA()  
train = pca.fit_transform(train)  
test = pca.transform(test) 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15, p = 2)
scores = cross_val_score(knn, train, Target, cv=10, scoring = f3_scorer)
cm = confusion_matrix(Target, y_pred)
print(cm)
proba = cross_val_predict(knn, train, Target, cv=10, method = 'predict_proba') 
fpr, tpr, _ = roc_curve(Target, proba[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print("Score F3 atingido:", scores.mean())
knn.fit(train,Target)
predict = knn.predict(test)
output =pd.DataFrame(Id)
output["ham"] = predict
output.to_csv("knn", index = False)