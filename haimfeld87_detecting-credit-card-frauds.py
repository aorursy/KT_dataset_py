import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/creditcard.csv")
df.columns
plt.figure(1,figsize=(10,8))

sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu")

plt.yticks(rotation=0) 
df=df.drop(['Time','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis=1)
X=df.drop(['Class'], axis=1)

Y=df['Class']
plt.figure()

Y.sort_index().value_counts().plot(kind='bar')

plt.ylabel('Counts')
RUS = RandomUnderSampler(random_state=0)

X_RUS ,Y_RUS = RUS.fit_sample(X, Y)
X_train, X_test,Y_train,Y_test = train_test_split(X_RUS,Y_RUS, test_size = 0.1, random_state=1245)
from pandas import Series

plt.figure()

Series(Y_train).value_counts().sort_index().plot(kind='bar')

plt.ylabel('Counts')
from sklearn.cross_validation import  cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC
models = [RandomForestClassifier(),KNeighborsClassifier(),XGBClassifier(),

          MLPClassifier(),LogisticRegression(),LinearSVC()]

names = ["RandomForestClassifier","KNeighborsClassifier","XGBClassifier",

          "MLPClassifier","LogisticRegression","LinearSVC"]



for model, name in zip(models, names):

    print (name)

    for score in ["accuracy","precision","recall"]:

        print (score)

        print (cross_val_score(model, X, Y,scoring=score, cv=3).mean() )

        print ("\n")
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

from itertools import cycle

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','black','pink', 'green'])





def roc_curve_acc(Y_test, Y_pred,method):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, color=next(colors),label='%s AUC = %0.3f'%(method, roc_auc))

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--')

    plt.xlim([-0.1,1.2])

    plt.ylim([-0.1,1.2])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')
from sklearn.metrics import confusion_matrix



RF=RandomForestClassifier()

RF.fit(X_train, Y_train)

Y_pred=RF.predict(X_test)

print("Random Forest Classifier report \n", classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,"RF")

print("Random Forest Classifier confusion matrix \n",confusion_matrix(Y_pred,Y_test))



KNN=KNeighborsClassifier()

KNN.fit(X_train, Y_train)

Y_pred=KNN.predict(X_test)

print("KNeighbors Classifier report \n",classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,'KNN')

print("KNeighbors Classifier confusion matrix \n",confusion_matrix(Y_pred,Y_test))



SVC=SVC()

SVC.fit(X_train, Y_train)

Y_pred=SVC.predict(X_test)

print("SVC Classifier report \n",classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,'SVC')

print("SVC Classifier confusion matrix \n",confusion_matrix(Y_pred,Y_test))



LSVC=LinearSVC()

LSVC.fit(X_train, Y_train)

Y_pred=LSVC.predict(X_test)

print("LSVC Classifier report \n",classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,'LSVC')

print("LSVC Classifier confusion matrix \n",confusion_matrix(Y_pred,Y_test))



XGB=XGBClassifier()

XGB.fit(X_train, Y_train)

Y_pred=XGB.predict(X_test)

print("XGB Classifier report \n",classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,'XGB')

print("XGB Classifier confusion matrix \n",confusion_matrix(Y_pred,Y_test))



MLP=MLPClassifier()

MLP.fit(X_train, Y_train)

Y_pred=MLP.predict(X_test)

print("MLP report \n",classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,'MLP')

print("MLP confusion matrix \n",confusion_matrix(Y_pred,Y_test))



LR=LogisticRegression()

LR.fit(X_train, Y_train)

Y_pred=LR.predict(X_test)

print("Logistic Regression report \n",classification_report(Y_pred,Y_test))

print('roc_auc_score: %0.3f'% roc_auc_score(Y_pred,Y_test))

roc_curve_acc(Y_test, Y_pred,'LR')

print("Logistic Regression confusion matrix \n",confusion_matrix(Y_pred,Y_test))