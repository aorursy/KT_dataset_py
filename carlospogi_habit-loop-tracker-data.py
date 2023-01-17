# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

import os

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/habit-loop-tracker/habitss.csv"  ,usecols=range(1,8))

df = df.replace(2, 1)
df.head()
df.columns

# Arthour is my late night activity which I focus on my personal projects for

# personal or professional development
df.describe()
X = df.iloc[:,[0,1,2,4,5,6]].values # other habits

y = df.iloc[:,3].values #arthour
#splitting the dataset into Training set and Test Set

from sklearn.model_selection import train_test_split

LR_X_train, LR_X_test, LR_y_train, LR_y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

KNN_X_train, KNN_X_test, KNN_y_train, KNN_y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

SVM_X_train, SVM_X_test, SVM_y_train, SVM_y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

NB_X_train, NB_X_test, NB_y_train, NB_y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

DT_X_train, DT_X_test, DT_y_train, DT_y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

RF_X_train, RF_X_test, RF_y_train, RF_y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

LR_classifier = LogisticRegression(random_state=0)

LR_classifier.fit(LR_X_train,LR_y_train)

LR_y_pred = LR_classifier.predict(LR_X_test)
from sklearn.neighbors import KNeighborsClassifier

KN_classifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p=2)

KN_classifier.fit(KNN_X_train,KNN_y_train)

KN_y_pred = KN_classifier.predict(KNN_X_test)
from sklearn.svm import SVC

SVC_classifier = SVC(kernel = 'linear', random_state=0)

SVC_classifier.fit(SVM_X_train,SVM_y_train)

SVC_y_pred = SVC_classifier.predict(SVM_X_test)
from sklearn.naive_bayes import GaussianNB

NB_classifier = GaussianNB()

NB_classifier.fit(NB_X_train,NB_y_train)

NB_y_pred = NB_classifier.predict(NB_X_test)
from sklearn.tree import DecisionTreeClassifier

DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

DT_classifier.fit(DT_X_train,DT_y_train)

DT_y_pred = DT_classifier.predict(DT_X_test)
from sklearn.ensemble import RandomForestClassifier

RF_classifier = RandomForestClassifier(n_estimators=10,criterion = 'entropy', random_state=0)

RF_classifier.fit(RF_X_train,RF_y_train)

RF_y_pred = RF_classifier.predict(RF_X_test)
from sklearn.model_selection import cross_val_score

#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

#LR

LR_CV_accuracies = cross_val_score(estimator = LR_classifier, X = LR_X_train, y = LR_y_train, cv = 10)

LR_CV_accuracy_mean = LR_CV_accuracies.mean()

LR_CV_accuracy_std = LR_CV_accuracies.std()



#KNN

KNN_CV_accuracies = cross_val_score(estimator = KN_classifier, X = KNN_X_train, y = KNN_y_train, cv = 10)

KNN_CV_accuracy_mean = KNN_CV_accuracies.mean()

KNN_CV_accuracy_std = KNN_CV_accuracies.std()



#SVM

SVM_CV_accuracies = cross_val_score(estimator = SVC_classifier, X = SVM_X_train, y = SVM_y_train, cv = 10)

SVM_CV_accuracy_mean = SVM_CV_accuracies.mean()

SVM_CV_accuracy_std = SVM_CV_accuracies.std()



#NB

NB_CV_accuracies = cross_val_score(estimator = NB_classifier, X = NB_X_train, y = NB_y_train, cv = 10)

NB_CV_accuracy_mean = LR_CV_accuracies.mean()

NB_CV_accuracy_std = LR_CV_accuracies.std()



#DT

DT_CV_accuracies = cross_val_score(estimator = DT_classifier, X = DT_X_train, y = DT_y_train, cv = 10)

DT_CV_accuracy_mean = DT_CV_accuracies.mean()

DT_CV_accuracy_std = DT_CV_accuracies.std()



#RF

RF_CV_accuracies = cross_val_score(estimator = RF_classifier, X = RF_X_train, y = RF_y_train, cv = 10)

RF_CV_accuracy_mean = RF_CV_accuracies.mean()

RF_CV_accuracy_std = RF_CV_accuracies.std()







from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score
#LR

# LR_accuracy = accuracy_score(LR_y_test, LR_y_pred)

LR_precision = precision_score(LR_y_test, LR_y_pred)

LR_recall = recall_score(LR_y_test, LR_y_pred)

LR_f1 = f1_score(LR_y_test, LR_y_pred)

LR_kappa = cohen_kappa_score(LR_y_test, LR_y_pred)



#KNN

# KNN_accuracy = accuracy_score(KNN_y_test,KN_y_pred)

KNN_precision = precision_score(KNN_y_test,KN_y_pred)

KNN_recall = recall_score(KNN_y_test,KN_y_pred)

KNN_f1 = f1_score(KNN_y_test,KN_y_pred)

KNN_kappa = cohen_kappa_score(KNN_y_test,KN_y_pred)



#SVM

# SVM_accuracy = accuracy_score(SVM_y_test,SVC_y_pred)

SVM_precision = precision_score(SVM_y_test,SVC_y_pred)

SVM_recall = recall_score(SVM_y_test,SVC_y_pred)

SVM_f1 = f1_score(SVM_y_test,SVC_y_pred)

SVM_kappa = cohen_kappa_score(SVM_y_test,SVC_y_pred)



#NB

# NB_accuracy = accuracy_score(NB_y_test,NB_y_pred)

NB_precision = precision_score(NB_y_test,NB_y_pred)

NB_recall = recall_score(NB_y_test,NB_y_pred)

NB_f1 = f1_score(NB_y_test,NB_y_pred)

NB_kappa = cohen_kappa_score(NB_y_test,NB_y_pred)



#DT

# DT_accuracy = accuracy_score(DT_y_test,DT_y_pred)

DT_precision = precision_score(DT_y_test,DT_y_pred)

DT_recall = recall_score(DT_y_test,DT_y_pred)

DT_f1 = f1_score(DT_y_test,DT_y_pred)

DT_kappa = cohen_kappa_score(DT_y_test,DT_y_pred)



#RF

# RF_accuracy = accuracy_score(RF_y_test,RF_y_pred)

RF_precision = precision_score(RF_y_test,RF_y_pred)

RF_recall = recall_score(RF_y_test,RF_y_pred)

RF_f1 = f1_score(RF_y_test,RF_y_pred)

RF_kappa = cohen_kappa_score(RF_y_test,RF_y_pred)



model_res1= {

    'LR':[LR_precision*100,LR_recall*100,LR_f1*100,LR_kappa*100],

    'SVM':[SVM_precision*100,SVM_recall*100,SVM_f1*100,SVM_kappa*100],

    'KNN':[KNN_precision*100,KNN_recall*100,KNN_f1*100,KNN_kappa*100],

    'NB':[NB_precision*100,NB_recall*100,NB_f1*100,NB_kappa*100],

    'DT':[DT_precision*100,DT_recall*100,DT_f1*100,DT_kappa*100],

    'RF':[RF_precision*100,RF_recall*100,RF_f1*100,RF_kappa*100]

}

eval1 = df = pd.DataFrame(model_res1, index =['Precision', 'Recall', 'F1','Kappa',]) 

eval1
model_res2= {

    'LR':[LR_CV_accuracy_mean*100,LR_CV_accuracy_std*100],

    'SVM':[SVM_CV_accuracy_mean*100,SVM_CV_accuracy_std],

    'KNN':[KNN_CV_accuracy_mean*100,KNN_CV_accuracy_std],

    'NB':[NB_CV_accuracy_mean*100,NB_CV_accuracy_std*100],

    'DT':[DT_CV_accuracy_mean*100,DT_CV_accuracy_std*100],

    'RF':[RF_CV_accuracy_mean*100,RF_CV_accuracy_std*100]

}

eval2 = df = pd.DataFrame(model_res2, index =['CV Accuracy Mean', 'CV Accuracy Std']) 

eval2
from sklearn.metrics import confusion_matrix
#LR

confusion_matrix(LR_y_test, LR_y_pred)
#KNN

confusion_matrix(KNN_y_test,KN_y_pred)
#SVM

confusion_matrix(SVM_y_test,SVC_y_pred)
#NB

confusion_matrix(NB_y_test,NB_y_pred)
#DT

confusion_matrix(DT_y_test,DT_y_pred)
#RF

confusion_matrix(RF_y_test,RF_y_pred)