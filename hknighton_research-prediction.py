import pandas as pd

import sklearn as skl

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn import preprocessing

from sklearn import metrics

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')







data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv") 





print(data.head())
def plot_roc(y_test, y_pred):

    fpr, tpr, thresholds = skl.metrics.roc_curve(y_test, y_pred, pos_label=1)

    roc_auc = skl.metrics.auc(fpr, tpr)

    plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area ={0:.2f})'.format(roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show();
len(data)
blanks = data.isna().sum()

len(data)

len(blanks)
#### split the data set

y = data['Research']

x = data.loc[:,:'CGPA']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



X_train.head()
from sklearn.ensemble import RandomForestClassifier , VotingClassifier 

from sklearn.linear_model import LogisticRegression 

from sklearn.svm import SVC
log_clf = LogisticRegression()

rnd_clf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

svm_clf = SVC(probability=True)
voting_clf = VotingClassifier(

    estimators = [('lr',log_clf), ('rf', rnd_clf), ('svc', svm_clf)],

    voting = 'soft')

voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

for clf in (log_clf,rnd_clf,svm_clf, voting_clf):

    clf.fit(X_train,y_train)

    y_prd = clf.predict(X_test)

    print(clf.__class__.__name__,accuracy_score(y_test,y_prd))

    plot_roc(y_test, y_prd)

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

auc(false_positive_rate, true_positive_rate)




