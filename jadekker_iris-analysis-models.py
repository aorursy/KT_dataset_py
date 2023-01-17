import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn import metrics

from sklearn.metrics import accuracy_score,make_scorer

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedKFold
#data import and preprocessing

dataset = pd.read_csv("../input/Iris.csv")

print(dataset.head())

lb_make = LabelEncoder()

dataset["Species"] = dataset["Species"].astype('category')

dataset["SepalRatio"] = np.divide(dataset["SepalLengthCm"],dataset["SepalWidthCm"])

dataset["PetalRatio"] = np.divide(dataset["PetalLengthCm"],dataset["PetalWidthCm"])

X_all = dataset[["Id","SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","SepalRatio","PetalRatio"]]

print(X_all.head())

y_all = lb_make.fit_transform(dataset["Species"])

acc_scorer = make_scorer(accuracy_score)
#training step LogReg

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()



parametersLR = {'C': [0.001,0.003,0.01,0.03,0.1,0.3,1]}

LRmodel = GridSearchCV(clf,parametersLR,scoring = acc_scorer)

_ = LRmodel.fit(X_all,y_all)
#Training step Naive Bayes

from sklearn.naive_bayes import GaussianNB

NBmodel = GaussianNB()

#no need to specify parameters for GaussianNB -> only priors considered

#_ = model.fit(X_train,y_train)

_ = NBmodel.fit(X_all,y_all)

#print(model)
#Training step kNN

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()



parametersKN = {'n_neighbors': [1,2,4,8,16]}

KNmodel = GridSearchCV(clf,parametersKN,scoring=acc_scorer)

_ = KNmodel.fit(X_all,y_all)
#Training step decision trees

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

parametersDT = {'criterion':["gini","entropy"],

            'max_depth': [1,2,3,4]}

DTmodel = GridSearchCV(clf,parametersDT,scoring=acc_scorer)

_ = DTmodel.fit(X_all,y_all)
#Training step SVM

from sklearn.svm import SVC

clf = SVC()



parametersSV = {'C': [0.001,0.003,0.01,0.03,0.1,0.3,1],

                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

SVmodel = GridSearchCV(clf,parametersSV,scoring=acc_scorer)

_ = SVmodel.fit(X_all,y_all)
#Training step XGBoost

from xgboost import XGBClassifier,plot_importance

clf = XGBClassifier()



parametersXG = {'n_estimators': [50,100,150,200],

               'max_depth': [2,4,6,8]}

kfold = StratifiedKFold(y_all,n_folds=10,shuffle=True,random_state=42)

XGmodel = GridSearchCV(clf,parametersXG,scoring=acc_scorer,n_jobs=-1,cv=kfold,verbose=1)

_ = XGmodel.fit(X_all,y_all)
#calculate model score

#expected = y_test

#predicted = model.predict(X_test)

expected = y_all



#Logistic Regression

LRpredicted = LRmodel.predict(X_all)

LRpredictions = lb_make.inverse_transform(LRpredicted)

LRpredictions = pd.DataFrame(LRpredictions)

print("The results for the Logistic Regression are:\n")

print(metrics.classification_report(expected,LRpredicted))

print(metrics.confusion_matrix(expected,LRpredicted))



#Naive Bayes

NBpredicted = NBmodel.predict(X_all)

NBpredictions = lb_make.inverse_transform(NBpredicted)

NBpredictions = pd.DataFrame(NBpredictions)

print("The results for the Naive Bayes are:\n")

print(metrics.classification_report(expected,NBpredicted))

print(metrics.confusion_matrix(expected,NBpredicted))



#kNN

KNpredicted = KNmodel.predict(X_all)

KNpredictions = lb_make.inverse_transform(KNpredicted)

KNpredictions = pd.DataFrame(KNpredictions)

print("The results for kNN are:\n")

print(metrics.classification_report(expected,KNpredicted))

print(metrics.confusion_matrix(expected,KNpredicted))



#Decision Tree

DTpredicted = DTmodel.predict(X_all)

DTpredictions = lb_make.inverse_transform(DTpredicted)

DTpredictions = pd.DataFrame(DTpredictions)

print("The results for the Decision tree are:\n")

print(metrics.classification_report(expected,DTpredicted))

print(metrics.confusion_matrix(expected,DTpredicted))



#SVM

SVpredicted = SVmodel.predict(X_all)

SVpredictions = lb_make.inverse_transform(SVpredicted)

SVpredictions = pd.DataFrame(SVpredictions)

print("The results for the support vector machine are:\n")

print(metrics.classification_report(expected,SVpredicted))

print(metrics.confusion_matrix(expected,SVpredicted))



#XGBoost

XGpredicted = XGmodel.predict(X_all)

XGpredictions = lb_make.inverse_transform(XGpredicted)

XGpredictions = pd.DataFrame(XGpredictions)

print("The results for the XGBoost are:\n")

print(metrics.classification_report(expected,XGpredicted))

print(metrics.confusion_matrix(expected,XGpredicted))





#Accuracies:

print("\nAcc. LogReg: {0}".format(accuracy_score(expected,LRpredicted)))

print("\nAcc. NaiveBayes: {0}".format(accuracy_score(expected,NBpredicted)))

print("\nAcc. kNN: {0}".format(accuracy_score(expected,KNpredicted)))

print("\nAcc. DecTree: {0}".format(accuracy_score(expected,DTpredicted)))

print("\nAcc. SVM: {0}".format(accuracy_score(expected,SVpredicted)))

print("\nAcc. XGBoost: {0}".format(accuracy_score(expected,XGpredicted)))
#output to a csv file

XGpredictions.to_csv('iris-predictions.csv',index=False)