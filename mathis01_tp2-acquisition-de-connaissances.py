import numpy as np



#Tout les imports - certains ne sont pas utilisés

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,accuracy_score

from sklearn.neural_network import MLPClassifier



import os

ds = open("../input/heart.csv")



# Initialisation des données



NB_EXEMPLES = 303

NB_ATTR = 13



# On fait deux tableaux de la taille de nos données initialisés à 0

X = np.zeros((NB_EXEMPLES, NB_ATTR))

Y = np.zeros(NB_EXEMPLES)



# on remplit nos tableaux avec nos données

for i,l in enumerate(ds):

    if i == 0: continue # on enlève la ligne descriptive

    t_l = l.rstrip('\r\n').split(',')

    for j,c in enumerate(t_l[0:13]):

        X[i-1,j] = float(c)

    Y[i-1] = float (t_l[-1])



# On définit les ensembles d'entrainement et de test

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,shuffle=True)



# On normalise les attributs pour leur donner le meme poids

scale = MinMaxScaler()

x_train = scale.fit_transform(x_train)

x_test = scale.transform(x_test)
# Random Forest

# On définit le classifieur : ici un random forest

rf = RandomForestClassifier(random_state=0)

# On donne les valeurs testées pour chacun des paramètres

param_rf={'n_estimators':[10,100,200,300,400], 'max_depth':[1,2,3,4,5]}

# On créé un classifieur qui teste toutes les combinaisons définies ci-dessus

# En plus de cela, le paramètre cv permet la cross-validation, et n_jobs parralellise les calculs.

clf = GridSearchCV(estimator=rf, param_grid=param_rf, cv=10, n_jobs=4, verbose=0, scoring='accuracy')



print ("Learn...")

clf.fit(x_train,y_train)

predictions = clf.predict(x_test)

print ("best n is:", clf.best_estimator_.n_estimators)

print ("best depth is:", clf.best_estimator_.max_depth)

print(classification_report(y_test, predictions,digits=4))
# Bayesian

# On définit un classifieur Bayesien

bay=GaussianNB()

# On donne les valeurs testées pour le lissage : ici 20 valeurs comprises entre 10e-20 et 10e-1

param_b = {'var_smoothing':np.logspace(-20, -1, num=20)}

clf = GridSearchCV(estimator=bay, param_grid=param_b, cv=10, n_jobs=4, verbose=0, scoring='accuracy')



print ("Learn...")

clf.fit(x_train,y_train);

predictions = clf.predict(x_test)

print ("best smoothing is:", clf.best_estimator_.var_smoothing)

print(classification_report(y_test, predictions,digits=4))

#SVM

# On définit un classifieur de machine à vecteurs de support

k = 'rbf' # noyau gaussien

svm = SVC(kernel=k,verbose=False) 



# On donne les valeurs testées pour chacun des paramètres

param_svm = {'gamma':np.logspace(-10, 10, num=10),'C':np.logspace(-10, 10, num=10)}

clf = GridSearchCV(estimator=svm, param_grid=param_svm,cv=10,n_jobs=4,verbose=0,scoring='accuracy')



print ("Learn...")

clf.fit(x_train,y_train);

predictions = clf.predict(x_test)

print ("best gamma is:", clf.best_estimator_.gamma)

print ("best C     is:", clf.best_estimator_.C)

print(classification_report(y_test, predictions,digits=4))

#knn

# On définit un classifieur aux k plus proches voisins

knn = KNeighborsClassifier()



# On mets dans un tableau les valeurs testées pour les plus proches voisins : entre 1 et 60

tab = []

for i in range (1,60):

    tab.append(i)



# On donne les valeurs testées pour chacun des paramètres

param_knn = {'n_neighbors':tab, 'algorithm' : ['ball_tree', 'kd_tree', 'brute']}

clf = GridSearchCV(estimator=knn, param_grid=param_knn, cv=10,n_jobs=4,verbose=0,scoring='accuracy')



print ("Learn...")

clf.fit(x_train,y_train)

print ("best k is:", clf.best_estimator_.n_neighbors)

print ("best algorithm is:", clf.best_estimator_.algorithm)

predictions = clf.predict(x_test)

print(classification_report(y_test, predictions,digits=4))
# neural network

# On créé un réseau de neurones

nn = MLPClassifier(hidden_layer_sizes=(15, 20)) 



# On définit plusieurs fonctions d'activation, et plusieurs optimisateurs

param_nn = {'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver' : ['lbfgs', 'sgd', 'adam']}

clf = GridSearchCV(estimator=nn, param_grid=param_nn, cv=10,n_jobs=4,verbose=0,scoring='accuracy')



print ("Learn...")

clf.fit(x_train,y_train);

predictions = clf.predict(x_test)

print ("best activation function is:", clf.best_estimator_.activation)

print ("best solver is:", clf.best_estimator_.solver)

print(classification_report(y_test, predictions,digits=4))