# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
X_data = pd.read_csv('../input/train.csv') #on cherche le dossier 
target = X_data['Survived'] #on supprime le survived pour être sur que notre algo ne connaisse pas les réponses
X_data = X_data.drop(['Name', 'Ticket', 'Embarked', 'PassengerId', 'Cabin'],  axis=1) #on supprime les colonnes peu porteuse d'informations
X_data.info() #on regarde si il y à des valeurs NaN
corr_matrix = X_data.corr()
corr_matrix #corrélation entre le prix et le taux de survie
corr_matrix["Survived"].sort_values(ascending=False) 
from	sklearn.preprocessing	import	LabelEncoder # on change les valeurs male et female en 0 et 1 pour pourvoir compiler les code
encoder	=	LabelEncoder()
X_cat_encoded = encoder.fit_transform(X_data['Sex'])
X_data = X_data.drop(['Sex'], axis = 1)
X_cat_encoded
X_data = pd.concat([X_data, pd.DataFrame(X_cat_encoded)], axis=1)
X_data.corr()
X_data = X_data.drop(['Survived'], axis=1)
from	sklearn.preprocessing	import	Imputer
imputer	=	Imputer(strategy="median")
imputer.fit(X_data)
X_data = imputer.transform(X_data)
X_data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_data,  target)
print(clf.feature_importances_) 
# le jeux de données n'est pas trés grand l'espace n'as pas l'ère d'être,linéairement séparable test svm
#méthode à noyaux
from	sklearn.svm	import	SVC
from	sklearn.preprocessing	import	StandardScaler
from	sklearn.model_selection	import	cross_val_score


scaler = StandardScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)

# svm_poly =SVC(kernel="poly", degree=3, coef0=1, C=5) demande énormément de puissance calculatoire
from	sklearn.model_selection	import	GridSearchCV
param_grid	=	[
                {
                'gamma': [0.01, 0.1, 0.05, 0,5, 0.2, 0.3 , 0.4, 0.6, 0.7, 1, 2, 3, 4, 5],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1100]},
              
		]
svm_clf = SVC()
grid_search	=	GridSearchCV(svm_clf,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_data,	target)
grid_search.best_params_
grid_search.best_estimator_
svm_clf = SVC(kernel="rbf", gamma=0.05, C=10)
scores  = cross_val_score(svm_clf,	X_data,	target,	cv=5,	scoring="accuracy")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
def validate(data, labels, clf):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from	sklearn.metrics	import	roc_auc_score
    
    i = 5
   
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc = []
    sss = StratifiedShuffleSplit(n_splits=5)
    for train_index, test_index in sss.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc.append(roc_auc_score(y_test, y_pred))
        
    print('Accuracy', np.mean(accuracy_scores))
    print('Precision', np.mean(precision_scores))
    print('Recall', np.mean(recall_scores))
    print('F1-measure', np.mean(f1_scores)) 
    print('Roc_AUC', np.mean(roc))
validate(X_data, target, svm_clf )

from	sklearn.linear_model	import	SGDClassifier
sgd_clf	=	SGDClassifier(random_state=42)
scores  = cross_val_score(sgd_clf,	X_data,	target,	cv=5,	scoring="accuracy")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
validate(X_data, target, sgd_clf )

from mlxtend.evaluate import paired_ttest_kfold_cv


t, p = paired_ttest_kfold_cv(estimator1=svm_clf,
                              estimator2=sgd_clf,
                              X=X_data, y=target,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
param_grid	=	[
                {'n_estimators' : [1, 10, 100, 200, 300, 500, 600, 700, 800, 900, 1000]},
              
		]
clf = RandomForestClassifier(n_jobs = -1)
grid_search	=	GridSearchCV(clf,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_data,	target)
grid_search.best_params_
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1, n_estimators=600)
scores  = cross_val_score(clf,	X_data,	target,	cv=5,	scoring="accuracy")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
validate(X_data, target, clf)

t, p = paired_ttest_kfold_cv(estimator1=clf,
                              estimator2=svm_clf,
                              X=X_data, y=target,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
param_grid	=	[
                { 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
              ]
grid_search	=	GridSearchCV(neigh,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_data,	target)
grid_search.best_params_
grid_search.best_estimator_
validate(X_data, target, neigh)

from mlxtend.evaluate import paired_ttest_kfold_cv


t, p = paired_ttest_kfold_cv(estimator1=svm_clf,
                              estimator2=neigh,
                              X=X_data, y=target,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
from	sklearn.linear_model	import	LogisticRegression
log_clf	=	LogisticRegression()
param_grid	=	[
                {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1100],
                 'class_weight' :[ 'balanced'],
                 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
              ]
grid_search	=	GridSearchCV(log_clf,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_data,	target)
grid_search.best_params_
log_clf	=	LogisticRegression(C = 1, class_weight = 'balanced')
validate(X_data, target, log_clf)

t, p = paired_ttest_kfold_cv(estimator1=svm_clf,
                              estimator2=log_clf,
                              X=X_data, y=target,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
from	sklearn.ensemble	import	VotingClassifier
log_clf	=	LogisticRegression(C = 1, class_weight = 'balanced')
clf = RandomForestClassifier(n_jobs=-1, n_estimators=600)
neigh = KNeighborsClassifier(n_neighbors = 5)
svm_clf = SVC(kernel="rbf", gamma=0.05, C=10)

voting_clf	=	VotingClassifier(
								estimators=[('lr',	log_clf),	('nb',	neigh),	('svc',	svm_clf)],
								voting='hard'
				)
validate(X_data, target, voting_clf)
from mlxtend.evaluate import paired_ttest_kfold_cv


t, p = paired_ttest_kfold_cv(estimator1=svm_clf,
                              estimator2=voting_clf,
                              X=X_data, y=target,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
clf.fit(X_data,  target)
print(clf.feature_importances_) 
X_feature = X_data[:, [0, 1, 4 ,5]]

from	sklearn.model_selection	import	GridSearchCV
param_grid	=	[
                {
                'gamma': [0.01, 0.1, 0.05, 0,5, 0.2, 0.3 , 0.4, 0.6, 0.7, 1, 2, 3, 4, 5],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1100]},
              
		]
svm_clf = SVC()
grid_search	=	GridSearchCV(svm_clf,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_feature,	target)
grid_search.best_params_
svm_clf = SVC(C = 10, gamma = 0.2)
validate(X_feature, target, svm_clf)
log_clf	=	LogisticRegression()
param_grid	=	[
                {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 1100],
                 'class_weight' :[ 'balanced'],
                 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
              ]
grid_search	=	GridSearchCV(log_clf,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_feature,	target)
grid_search.best_params_
log_clf	=	LogisticRegression(C = 1, class_weight = 'balanced')
validate(X_feature, target, log_clf )
neigh = KNeighborsClassifier()
param_grid	=	[
                { 'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
              ]
grid_search	=	GridSearchCV(neigh,	param_grid,	cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_feature,	target)
grid_search.best_params_
neigh = KNeighborsClassifier(n_neighbors = 4)
validate(X_feature, target, neigh)
voting_clf	=	VotingClassifier(
								estimators=[('lr',	log_clf),	('nb',	neigh),	('svc',	svm_clf)],
								voting='hard'
				)
validate(X_data, target, voting_clf)


