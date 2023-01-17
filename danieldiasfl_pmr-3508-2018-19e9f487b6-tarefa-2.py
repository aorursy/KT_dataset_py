import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

import os
print(os.listdir("../input"))

#sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
train = pd.read_csv("../input/dataset/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

test = pd.read_csv("../input/dataset/test_features.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.shape
train.describe()
train.sample(5)
spam = train[train.ham == False]
ham = train[train.ham == True]
spam.describe()
ham.describe()
spam_sum = spam.drop(columns = ['Id', 'ham']).sum().sort_values(ascending = False)
ham_sum = ham.drop(columns = ['Id', 'ham']).sum().sort_values(ascending = False)
spam_sum.iloc[3:56]
ham_sum.iloc[3:56]
atr = []
for i in range(0,56):
    for j in range(i,56):
        if(spam_sum.index[i] == ham_sum.index[j] and abs(i - j) > 25):
            atr.append(spam_sum.index[i])
            
print (atr)
remove = ([spam_sum[atr[0]], ham_sum[atr[0]]])
business = ([spam_sum[atr[1]], ham_sum[atr[1]]])
word000 = ([spam_sum[atr[2]], ham_sum[atr[2]]])
internet = ([spam_sum[atr[3]], ham_sum[atr[3]]])
money = ([spam_sum[atr[4]], ham_sum[atr[4]]])
credit = ([spam_sum[atr[5]], ham_sum[atr[5]]])
char =  ([spam_sum[atr[6]], ham_sum[atr[6]]])

ax = ['spam', 'ham']
plt.figure(1,figsize = (15,3))
plt.subplot(171)
plt.bar(ax, remove)
plt.title('remove')
plt.subplot(172)
plt.bar(ax, business)
plt.title('business')
plt.subplot(173)
plt.bar(ax, word000)
plt.title('000')
plt.subplot(174)
plt.bar(ax, internet)
plt.title('internet')
plt.subplot(175)
plt.bar(ax, money)
plt.title('money')
plt.subplot(176)
plt.bar(ax, credit)
plt.title('credit')
plt.subplot(177)
plt.bar(ax, char)
plt.title('$')
Xtrain = train[["word_freq_remove","word_freq_business","word_freq_000","word_freq_internet","word_freq_money","word_freq_credit",
                "char_freq_$"]]
Ytrain = train['ham']

Xtest = test[['word_freq_remove','word_freq_business','word_freq_000','word_freq_internet','word_freq_money','word_freq_credit',
                'char_freq_$']]
gnb = GaussianNB()
y_pred = gnb.fit(Xtrain, Ytrain).predict(Xtest)
score = cross_val_score(gnb, Xtrain, Ytrain, cv = 10)
print (score)
print(score.mean())
#exemplo de matriz de confusão
test_cm = Xtrain[0:400]
Ytest_cm = Ytrain[0:400]
train_cm = Xtrain[400:3280]
Ytrain_cm = Ytrain[400:3280]
ycm_pred = gnb.fit(train_cm, Ytrain_cm).predict(test_cm)

confusion_matrix(Ytest_cm, ycm_pred)
m = 0
mv = 0
n = []
for i in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(Xtrain, Ytrain)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    k = np.average(scores)
    n.append(k)
    
pred = knn.fit(Xtrain, Ytrain).predict(Xtest)
plt.plot(n)
knn = KNeighborsClassifier(n_neighbors=10)
ycm_pred_knn = knn.fit(train_cm, Ytrain_cm).predict(test_cm)
confusion_matrix(Ytest_cm, ycm_pred_knn)

predf = []
for i in range(0,921):
    predf.append([test['Id'][i],pred[i]])
    
for i in range(0,5):
    print(predf[i])
out = pd.DataFrame(predf, columns = ['Id', 'ham'])
out.to_csv("pred.csv", index = False)
out