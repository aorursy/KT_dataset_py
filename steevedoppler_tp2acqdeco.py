# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")
import codecs

Y=np.zeros(data.shape[0])

X=np.zeros((data.shape[0], 13))

for i,l in enumerate(codecs.open("../input/heart.csv", "r", "utf-8")):

    if i!=0:

        t_l=l.rstrip('\r\n').split(',')

        Y[i-1]=t_l[-1]

        for j,a in enumerate(t_l[0:-2]):

            X[i-1,j]=float(a)

            

# Pour chaque classifieur, la validation croisée a été utilisée
# Random Forest !

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold



clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

kf=KFold(n_splits=20,shuffle=True)



# Validation croisée

score=0

# 20 séparation en training set et test set

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

    

print('average accuracy:',score/20)

print('empiric accuracy is',clf.score(X,Y))



# Etude sur l'importance de chaque feature

impo = clf.feature_importances_



plt.rcdefaults()

fig, ax = plt.subplots()



attributs = ('age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal')

y_pos = np.arange(len(attributs))



ax.barh(y_pos, impo, align='center', color='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(attributs)

ax.invert_yaxis()  

ax.set_xlabel('Importances')

ax.set_title('Feature importances')



plt.show()
from sklearn import neighbors



nb_neighb = 3

clf = neighbors.KNeighborsClassifier(nb_neighb) 

kf=KFold(n_splits=20,shuffle=True)



# Validation croisée

score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

print('Nombre de voisins:', nb_neighb)

print('average accuracy:',score/20)

print('empiric accuracy:',clf.score(X,Y))
from sklearn import neighbors



nb_neighb = 8

clf = neighbors.KNeighborsClassifier(nb_neighb) 

kf=KFold(n_splits=20,shuffle=True)



# Validation croisée

score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

print('Nombre de voisins:', nb_neighb)

print('average accuracy:',score/20)

print('empiric accuracy:',clf.score(X,Y))
from sklearn import neighbors



nb_neighb = 20

clf = neighbors.KNeighborsClassifier(nb_neighb)

kf=KFold(n_splits=20,shuffle=True)



# Validation croisée

score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

print('Nombre de voisins:', nb_neighb)

print('average accuracy:',score/20)

print('empiric accuracy:',clf.score(X,Y))
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

kf=KFold(n_splits=20,shuffle=True)



# Cette méthode a le meilleur score sans compter la RandomForest

# Validation croisée

score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    gnb.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + gnb.score(X_test,Y_test)

    

print('average accuracy:', score/20)

print('empiric accuracy:', gnb.score(X,Y))
from sklearn.svm import SVC



# Noyau rbf

clf = SVC(gamma='auto', kernel='rbf')

kf=KFold(n_splits=20,shuffle=True)



# Validation croisée

score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

    

# Ce classifieur est en surapprentissage

print('average accuracy:', score/20)

print('empiric accuracy:', clf.score(X,Y))
from sklearn.svm import SVC



# Cette méthode prend trop de temps à être éxécuter

# Elle est commentée pour valider le commit

"""

clf = SVC(kernel='poly', gamma='auto', degree=3)

kf=KFold(n_splits=20,shuffle=True)



score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

    

print('average accuracy:', score/20)

print('empiric accuracy is', clf.score(X,Y))

"""
from sklearn.svm import SVC



clf = SVC(kernel='linear')

# Le noyeu linéaire marche mieux que le noyau rbf, qui sur-apprend

kf=KFold(n_splits=20,shuffle=True)



score=0

for training_ind,test_ind in kf.split(X):

    X_train=X[training_ind]

    Y_train=Y[training_ind]

    clf.fit(X_train, Y_train)

    X_test=X[test_ind]

    Y_test=Y[test_ind]

    score = score + clf.score(X_test,Y_test)

    

print('average accuracy:', score/20)

print('empiric accuracy is', clf.score(X,Y))