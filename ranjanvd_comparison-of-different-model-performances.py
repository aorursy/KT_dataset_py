# Importing required libraries



%matplotlib inline

# import necessary libraries and specify that graphs should be plotted inline. 

from sklearn.datasets import load_iris

from sklearn import tree

from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV,KFold

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix,roc_curve, auc,matthews_corrcoef

from matplotlib.legend_handler import HandlerLine2D

from sklearn.preprocessing import Normalizer,MinMaxScaler

import scikitplot as skplt

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')
# Loading the data set from sklearn. As we won't be using Id column, we can use data file from sklearn.

# Further, both sklearn and the given data are same.



from sklearn.datasets import load_breast_cancer

wdbc = load_breast_cancer()

print(wdbc.DESCR)
## Explore the data set

n_samples, n_features = wdbc.data.shape



#print(type(wdbc))

#print(wdbc.keys())



print ('The dimensions of the data set are', n_samples, 'by', n_features)

print('*'*75)

print('The classes are: ', wdbc.target_names)

print('*'*75)

print('The features in the data set are:', wdbc.feature_names)

## Explore the data set

print('Data:',wdbc.data[:2])

print('*'*75)

print('Target:',wdbc.target[:2])
##************************************************************************************

## 

## DECISION TREE

## 

##************************************************************************************

train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]



# Splitting data into train and test in 80-20 ratio.

X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2)



max_depths = range(1,30) 

min_leaf_size = range(1,30)



for depth in max_depths:

    # Decision tree for varying depth

    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=depth)

    

    ## ******************************************************

    # We can use AUC-ROC curve to select our hyperparameter.

    # Here, ROC is a probability curve and AUC represents degree of separability.

    # It tells how much model is capable of distinguishing between True and Flase output.

    # Higher the AUC means the model is better at predicting 0s as 0s and 1s as 1s.

    ## ******************************************************

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results  

}

depths = pd.DataFrame(dict(mydict),index=max_depths)



train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]



for leaf in min_leaf_size:

    # Decision tree for varying depth

    clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_leaf=leaf)

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results

}

leafs = pd.DataFrame(dict(mydict),index=min_leaf_size)    

# skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)

train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]



for impurity in [.0001,.001,.01,.1,1,10,]:

    # Decision tree for varying depth

    clf = tree.DecisionTreeClassifier(criterion="gini", min_impurity_decrease=impurity)

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results

    

}

impurity = pd.DataFrame(dict(mydict),index=[.0001,.001,.01,.1,1,10,])  



## ***************************************************************************************



fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(231)

ax3 = fig.add_subplot(232)

ax5 = fig.add_subplot(233)

ax2 = fig.add_subplot(234)

ax4 = fig.add_subplot(235)

ax6 = fig.add_subplot(236)



ax5.set_xscale('log')

ax6.set_xscale('log')



depths[['Train accuracies','Test accuracies']].plot(ax=ax1)

depths[['AUC train results','AUC test results']].plot(ax=ax2)

leafs[['Train accuracies','Test accuracies']].plot(ax=ax3)

leafs[['AUC train results','AUC test results']].plot(ax=ax4)

impurity[['Train accuracies','Test accuracies']].plot(ax=ax5)

impurity[['AUC train results','AUC test results']].plot(ax=ax6)

ax1.title.set_text('Accuracy vs Tree Depth')

ax2.title.set_text('AUC vs Tree Depth')

ax3.title.set_text('Accuracy vs Min leaf size')

ax4.title.set_text('AUC vs min leaf size')

ax5.title.set_text('Accuracy vs Min impurity decrease')

ax6.title.set_text('AUC vs vs Min impurity decrease')

plt.show()



print('*'*100,'\n')

print('Using GridSearchCV fidning the best model')

# We use Grid search to optimize mutiple paramers:

parameters = {'min_impurity_decrease':[.01,.1,0,1,10], 'max_depth':list(range(1,10)),'min_samples_leaf':[1,5,10,15,20]}

dtree = tree.DecisionTreeClassifier()

clf = GridSearchCV(dtree, parameters, cv=7,scoring='f1')

clf.fit(wdbc.data, wdbc.target)

print('The best hyperparameters are:',clf.best_params_)

print('The best score is:',clf.best_score_)
# Final Model

clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=9,min_samples_leaf=5,min_impurity_decrease=0)

clf = clf.fit(wdbc.data, wdbc.target)

scores = cross_val_score(clf, wdbc.data, wdbc.target, cv=7,scoring='f1')

print(scores)

# The mean score and the 95% confidence interval of the score estimate are hence given by:

print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



print('*'*100)



X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2,random_state = 1)

clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

probas = clf.predict_proba(X_test)

print(classification_report(y_test,predictions))

print('*'*100)

print('Confusion Matrix\n',confusion_matrix(y_test, predictions))

print('*'*100)

print('Matthews Corrcoef',matthews_corrcoef(y_test,predictions))

print('*'*100)
## ******************************************************

# We can use AUC-ROC curve to select our hyperparameter.

# Here, ROC is a probability curve and AUC represents degree of separability.

# It tells how much model is capable of distinguishing between True and Flase output.

# Higher the AUC means the model is better at predicting 0s as 0s and 1s as 1s.

## ******************************************************

    

fig = plt.figure(figsize=(15,4))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)

skplt.metrics.plot_precision_recall_curve(y_test, probas,ax=ax1)

skplt.metrics.plot_roc(y_test, probas,ax=ax2)

skplt.metrics.plot_lift_curve(y_test, probas,ax=ax3)

plt.show()
##************************************************************************************

## 

## Logistic regression

## 

##************************************************************************************



c_set = range(1,20)



# Splitting data

X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2,random_state=1)



train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]

for c in c_set:

    # logit with varying c value (Inverse of regularization strength)

    logreg = LogisticRegression(C=c, penalty='l1')

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results

}



print('Logistic Regression for non scaled data')

c_change_non_normal = pd.DataFrame(dict(mydict),index=c_set)  

logreg = LogisticRegression(C=5, penalty='l1')

scores = cross_val_score(logreg, wdbc.data, wdbc.target, cv=7,scoring='f1')

print(scores)

# The mean score and the 95% confidence interval of the score estimate are hence given by:

print("\nF1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# ********************************************************************************************



# Scaling the data

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]

for c in c_set:

    # logit with varying c value (Inverse of regularization strength)

    logreg = LogisticRegression(C=c, penalty='l2')

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results

}

c_change_normal = pd.DataFrame(dict(mydict),index=c_set)  



# ********************************************************************************************

print('*'*100)

print('Logistic Regression for scaled data')

logreg = LogisticRegression(C=5, penalty='l2')

scores = cross_val_score(logreg, wdbc.data, wdbc.target, cv=7,scoring='f1')

print(scores)

# The mean score and the 95% confidence interval of the score estimate are hence given by:

print("\nF1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('*'*100)



train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]

for  p in ['l1','l2']:

    # logit with varying c value (Inverse of regularization strength)

    logreg = LogisticRegression(penalty=p)

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results,

}



penalties  = pd.DataFrame(dict(mydict),index=['l1','l2'])



## ***************************************************************************************

fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(231)

ax3 = fig.add_subplot(232)

ax5 = fig.add_subplot(233)

ax2 = fig.add_subplot(234)

ax4 = fig.add_subplot(235)

ax6 = fig.add_subplot(236)



ax5.set_xscale('log')

ax6.set_xscale('log')



c_change_non_normal[['Train accuracies','Test accuracies']].plot(ax=ax1)

c_change_non_normal[['AUC train results','AUC test results']].plot(ax=ax2)

c_change_normal[['Train accuracies','Test accuracies']].plot(ax=ax3)

c_change_normal[['AUC train results','AUC test results']].plot(ax=ax4)

penalties[['Train accuracies','Test accuracies']].plot(ax=ax5,kind = 'barh')

penalties[['AUC train results','AUC test results']].plot(ax=ax6,kind = 'barh')

ax1.title.set_text('Accuracy vs C(Inverse of regularization strength) Non Scaled Data')

ax2.title.set_text('AUC vs C(Inverse of regularization strength) Non Scaled Data')

ax3.title.set_text('Accuracy vs C(Inverse of regularization strength) Scaled Data')

ax4.title.set_text('AUC vs min C(Inverse of regularization strength) Scaled Data')

ax5.title.set_text('Accuracy vs Penality')

ax6.title.set_text('AUC vs vs Penality')

plt.show()
# GRIDSEARCH

print('*'*100,'\n')

print('Using GridSearchCV fidning the best model')

# We use Grid search to optimize mutiple paramers:

parameters = {'C':range(1,10), 'penalty':['l1','l2']}

logit = LogisticRegression()

clf = GridSearchCV(logit, parameters, cv=7, scoring='f1')

clf.fit(wdbc.data, wdbc.target)

print('The best hyperparameters are:',clf.best_params_)

print('The best score is:',clf.best_score_)
# Splitting data

X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2,random_state = 1)



# Final Model - LOGIT

clf = LogisticRegression(C=8, penalty='l1')

scores = cross_val_score(logreg, wdbc.data, wdbc.target, cv=7,scoring='f1')

print(scores)

# The mean score and the 95% confidence interval of the score estimate are hence given by:

print("\nF1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



print('*'*100)



X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2,random_state = 1)

clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

probas = clf.predict_proba(X_test)

print(classification_report(y_test,predictions))

print('*'*100)

print('Confusion Matrix\n',confusion_matrix(y_test, predictions))

print('*'*100)

print('Matthews Corrcoef',matthews_corrcoef(y_test,predictions))
fig = plt.figure(figsize=(15,4))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)

skplt.metrics.plot_precision_recall_curve(y_test, probas,ax=ax1)

skplt.metrics.plot_roc(y_test, probas,ax=ax2)

skplt.metrics.plot_lift_curve(y_test, probas,ax=ax3)

plt.show()
##******************************************************

## 

## KNN

## 

##******************************************************



# Optimize KNN classifier and detect (potential) over-fitting



neighbors_values = range(1,30)

scaler = MinMaxScaler()



X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2)

# Scaling the features to fit it into KNN

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



cross_validation_score = []

for n in neighbors_values:

    # KNN with varying n value

    knn = KNeighborsClassifier(n_neighbors = n)

    # Cross validation scrore

    cross_validation_score.append(cross_val_score(knn, wdbc.data, wdbc.target, cv=7,scoring = 'f1').mean())



mydict_ = {

    'Cross validation score':cross_validation_score

}



cross_validation_score = []

for wt in ["uniform", "distance"]:

    # KNN with varying n value

    knn = KNeighborsClassifier(weights = wt)

    # Cross validation scrore

    cross_validation_score.append(cross_val_score(knn, wdbc.data, wdbc.target, cv=7,scoring = 'f1').mean())

    

mydict = {

    'Cross validation score':cross_validation_score

}



neighbors = pd.DataFrame(dict(mydict_),index=neighbors_values)  

weights = pd.DataFrame(dict(mydict),index=["uniform", "distance"])  



fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)





neighbors[['Cross validation score']].plot(ax=ax1)

weights[['Cross validation score']].plot(kind='bar',ax=ax2)

ax1.title.set_text('F1 Score vs Neighbours')

ax2.title.set_text('AUC vs Weights')

plt.show()

# GRIDSEARCH

scaler = MinMaxScaler()

scaler.fit(wdbc.data)

x = scaler.transform(wdbc.data)

print('*'*100,'\n')

print('Using GridSearchCV fidning the best model')

# We use Grid search to optimize mutiple paramers:

parameters = {'weights':["uniform", "distance"], 'n_neighbors' : range(5,15)}

knn = KNeighborsClassifier()

clf = GridSearchCV(knn, parameters, cv=7,scoring='f1')

clf.fit(x, wdbc.target)

print('The best hyperparameters are:',clf.best_params_)

print('The best score is:',clf.best_score_)
# Final Model - KNN

knn = KNeighborsClassifier(n_neighbors= 12, weights= 'distance')

scores = cross_val_score(knn, wdbc.data, wdbc.target, cv=7,scoring='f1')

print(scores)

# The mean score and the 95% confidence interval of the score estimate are hence given by:

print("\nF1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2,random_state = 2)

# Scaling the features to fit it into KNN

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

knn = knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

probas = knn.predict_proba(X_test)

print('*'*100)

print(classification_report(y_test,predictions))

print('*'*100)

print('Confusion Matrix\n',confusion_matrix(y_test, predictions))

print('*'*100)

print('Matthews Corrcoef',matthews_corrcoef(y_test,predictions))
fig = plt.figure(figsize=(15,4))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)

skplt.metrics.plot_precision_recall_curve(y_test, probas,ax=ax1)

skplt.metrics.plot_roc(y_test, probas,ax=ax2)

skplt.metrics.plot_lift_curve(y_test, probas,ax=ax3)

plt.show()
##************************************************************************************

## 

## SVM

## 

##************************************************************************************

train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]



# Splitting data into train and test in 80-20 ratio.

X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2)



scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



C = range(1,30) 

gamma = [.001,.01,.1,1,10,100]



for i in C:

    # Decision tree for varying depth

    clf = SVC(kernel="linear", C=i,probability=True)

    

    ## ******************************************************

    # We can use AUC-ROC curve to select our hyperparameter.

    # Here, ROC is a probability curve and AUC represents degree of separability.

    # It tells how much model is capable of distinguishing between True and Flase output.

    # Higher the AUC means the model is better at predicting 0s as 0s and 1s as 1s.

    ## ******************************************************

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results  

}

depths = pd.DataFrame(dict(mydict),index=C)



train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]



for i in C:

    # Decision tree for varying depth

    clf = SVC(kernel="rbf", C=i,probability=True)

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results

}

leafs = pd.DataFrame(dict(mydict),index=C)    

# skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)

train_accuracies,test_accuracies = [],[]

auc_train_results,auc_test_results = [],[]



for g in gamma:

    # Decision tree for varying depth

    clf = SVC(kernel="rbf", gamma=g,probability=True)

    

    # auc score for training data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, clf.fit(X_train, y_train).predict_proba(X_train)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_train_results.append(roc_auc)

    

    # auc score for testing data

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.fit(X_train, y_train).predict_proba(X_test)[:,1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    auc_test_results.append(roc_auc)

    

    # accuracy for testing data

    test_accuracies.append(clf.fit(X_train, y_train).score(X_test, y_test))

    

    # accuracy for tarining data

    train_accuracies.append(clf.fit(X_train, y_train).score(X_train, y_train))

    

mydict = {

    'Train accuracies':train_accuracies,\

    'Test accuracies':test_accuracies,\

    'AUC train results':auc_train_results,\

    'AUC test results':auc_test_results

    

}

impurity = pd.DataFrame(dict(mydict),index=gamma)  



## ***************************************************************************************



fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(231)

ax3 = fig.add_subplot(232)

ax5 = fig.add_subplot(233)

ax2 = fig.add_subplot(234)

ax4 = fig.add_subplot(235)

ax6 = fig.add_subplot(236)



ax5.set_xscale('log')

ax6.set_xscale('log')





depths[['Train accuracies','Test accuracies']].plot(ax=ax1)

depths[['AUC train results','AUC test results']].plot(ax=ax2)

leafs[['Train accuracies','Test accuracies']].plot(ax=ax3)

leafs[['AUC train results','AUC test results']].plot(ax=ax4)

impurity[['Train accuracies','Test accuracies']].plot(ax=ax5)

impurity[['AUC train results','AUC test results']].plot(ax=ax6)

ax1.title.set_text('Accuracy vs C - Linear SVM')

ax2.title.set_text('AUC vs C - Linear SVM')

ax3.title.set_text('Accuracy vs C - Non Linear SVM')

ax4.title.set_text('AUC vs C - Non Linear SVM')

ax5.title.set_text('Accuracy vs Gamma - Non Linear SVM')

ax6.title.set_text('AUC vs Gamma - Non Linear SVM')

plt.show()
# GRIDSEARCH

scaler = MinMaxScaler()

scaler.fit(wdbc.data)

X = scaler.transform(wdbc.data)



print('*'*100,'\n')

print('Using GridSearchCV fidning the best model for Linear SVC')

# We use Grid search to optimize mutiple paramers:

parameters = {'kernel':['linear','rbf'],'C':range(1,30)}

clf = SVC()

clf = GridSearchCV(clf, parameters, cv=7, scoring='f1')

clf.fit(X, wdbc.target)

print('The best hyperparameters are:',clf.best_params_)

print('The best score is:',clf.best_score_)





print('*'*100,'\n')

print('Using GridSearchCV fidning the best model for Non-Linear SVC')

parameters = {'kernel':['rbf','sigmoid'],'C':range(1,30),'gamma':[.001,.01,.1,1,10,100]}

clf = SVC()

clf = GridSearchCV(clf, parameters, cv=7, scoring='f1')

clf.fit(X, wdbc.target)

print('The best hyperparameters are:',clf.best_params_)

print('The best score is:',clf.best_score_)
# Final Model

clf = SVC(kernel="linear", C=4,probability=True)

clf = clf.fit(wdbc.data, wdbc.target)

scores = cross_val_score(clf, X, wdbc.target, cv=7,scoring='f1')

print(scores)

# The mean score and the 95% confidence interval of the score estimate are hence given by:

print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



print('*'*100)



X_train, X_test, y_train, y_test = train_test_split(wdbc.data, wdbc.target, test_size=0.2,random_state = 1)



scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



clf = clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

probas = clf.predict_proba(X_test)

print(classification_report(y_test,predictions))

print('*'*100)

print('Confusion Matrix\n',confusion_matrix(y_test, predictions))

print('*'*100)

print('Matthews Corrcoef',matthews_corrcoef(y_test,predictions))

print('*'*100)
fig = plt.figure(figsize=(15,4))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)

skplt.metrics.plot_precision_recall_curve(y_test, probas,ax=ax1)

skplt.metrics.plot_roc(y_test, probas,ax=ax2)

skplt.metrics.plot_lift_curve(y_test, probas,ax=ax3)

plt.show()