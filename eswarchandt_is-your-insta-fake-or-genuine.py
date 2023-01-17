import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
insta_train=pd.read_csv(r"../input/instagram-fake-spammer-genuine-accounts/train.csv")

insta_test=pd.read_csv(r"../input/instagram-fake-spammer-genuine-accounts/test.csv")

insta_train.head()
insta_train.describe()
insta_train.info()
print(insta_train.shape)

print(insta_test.shape)
print(insta_train.isna().values.any().sum())

print(insta_test.isna().values.any().sum())
corr=insta_train.corr()

sns.heatmap(corr)

sns.pairplot(insta_train)
train_Y=insta_train.fake

train_Y=pd.DataFrame(train_Y)

train_Y.tail(12)
train_X= insta_train.drop(columns='fake')

train_X.head()
test_Y=insta_test.fake

test_Y=pd.DataFrame(test_Y)

test_Y.tail(12)
test_X= insta_test.drop(columns='fake')

test_X.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
logreg=LogisticRegression()

model_1=logreg.fit(train_X,train_Y)

logreg_predict= model_1.predict(test_X)
accuracy_score(logreg_predict,test_Y)
print(classification_report(logreg_predict,test_Y))
def plot_confusion_matrix(test_Y, predict_y):

 C = confusion_matrix(test_Y, predict_y)

 A =(((C.T)/(C.sum(axis=1))).T)

 B =(C/C.sum(axis=0))

 plt.figure(figsize=(20,4))

 labels = [1,2]

 cmap=sns.light_palette("blue")

 plt.subplot(1, 3, 1)

 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

 plt.xlabel('Predicted Class')

 plt.ylabel('Original Class')

 plt.title("Confusion matrix")

 plt.subplot(1, 3, 2)

 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

 plt.xlabel('Predicted Class')

 plt.ylabel('Original Class')

 plt.title("Precision matrix")

 plt.subplot(1, 3, 3)

 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

 plt.xlabel('Predicted Class')

 plt.ylabel('Original Class')

 plt.title("Recall matrix")

 plt.show()
plot_confusion_matrix(test_Y, logreg_predict)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

model_2= knn.fit(train_X,train_Y)

knn_predict=model_2.predict(test_X)
accuracy_score(knn_predict,test_Y)

print(classification_report(test_Y,knn_predict))
plot_confusion_matrix(test_Y, knn_predict)
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

model_3=dtree.fit(train_X,train_Y)

dtree_predict=model_3.predict(test_X)
accuracy_score(dtree_predict,test_Y)

print(classification_report(dtree_predict,test_Y))
plot_confusion_matrix(test_Y, dtree_predict)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

model_4=rfc.fit(train_X,train_Y)

rfc_predict=model_4.predict(test_X)
accuracy_score(rfc_predict,test_Y)
print(classification_report(rfc_predict,test_Y))
plot_confusion_matrix(test_Y, rfc_predict)
from sklearn.svm import SVC

svc=SVC()

model_5=svc.fit(train_X,train_Y)

svm_predict=model_5.predict(test_X)
accuracy_score(svm_predict,test_Y)
print(classification_report(svm_predict,test_Y))
plot_confusion_matrix(test_Y,svm_predict)
from sklearn.ensemble import AdaBoostClassifier

adc=AdaBoostClassifier(n_estimators=5,learning_rate=1)

model_6=adc.fit(train_X,train_Y)

adc_predict=model_6.predict(test_X)
accuracy_score(adc_predict,test_Y)
print(classification_report(adc_predict,test_Y))
plot_confusion_matrix(test_Y,adc_predict)
from xgboost import XGBClassifier

xgb=XGBClassifier()

model_7=xgb.fit(train_X,train_Y)

xgb_predict=model_7.predict(test_X)
accuracy_score(xgb_predict,test_Y)
print(classification_report(xgb_predict,test_Y))
plot_confusion_matrix(test_Y, xgb_predict)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

kfold = StratifiedKFold(n_splits=10)

# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid,scoring="accuracy", verbose = 1,cv=kfold)



gsadaDTC.fit(train_X,train_Y)



gsadaDTC_model=gsadaDTC.predict(test_X)

accuracy_score(gsadaDTC_model,test_Y)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid,scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(train_X,train_Y)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
gsExtC_model=gsExtC.predict(test_X)

accuracy_score(gsExtC_model,test_Y)
RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(train_X,train_Y)

gsRFC_model=gsRFC.predict(test_X)

accuracy_score(gsRFC_model,test_Y)
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(train_X,train_Y)

gsGBC_model=gsGBC.predict(test_X)

accuracy_score(gsGBC_model,test_Y)
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(train_X,train_Y)

gsSVMC_model=gsSVMC.predict(test_X)

accuracy_score(gsSVMC_model,test_Y)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",train_X,train_Y,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",train_X,train_Y,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",train_X,train_Y,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",train_X,train_Y,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",train_X,train_Y,cv=kfold)

print('Logistic Regression Accuracy:',accuracy_score(logreg_predict,test_Y))

print('K-Nearest Neighbour Accuracy:',accuracy_score(knn_predict,test_Y))

print('Decision Tree Classifier Accuracy:',accuracy_score(dtree_predict,test_Y))

print('Random Forest Classifier Accuracy:',accuracy_score(rfc_predict,test_Y))

print('support Vector Machine Accuracy:',accuracy_score(svm_predict,test_Y))

print('Adaboost Classifier Accuracy:',accuracy_score(adc_predict,test_Y))

print('XGBoost Accuracy:',accuracy_score(xgb_predict,test_Y))

print('SVC:',accuracy_score(gsSVMC_model,test_Y))

print('Gradient Boosting Classifier:',accuracy_score(gsGBC_model,test_Y))

print('RF classifier:',accuracy_score(gsRFC_model,test_Y))

print('Extensible Trees Classifier:',accuracy_score(gsExtC_model,test_Y))

print('Adaboost Classifier:',accuracy_score(gsadaDTC_model,test_Y))
